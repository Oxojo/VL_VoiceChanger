"""Untrained network/state tests; these do not establish voice quality."""
import unittest
import torch

from engine.experimental.streamvc import CausalConv, CausalTranspose, StreamVCNeuralCore


class TestStreamVCPrimitives(unittest.TestCase):
    @torch.inference_mode()
    def test_strided_dilated_conv_arbitrary_boundaries(self):
        for stride, dilation in [(1, 1), (2, 1), (4, 3), (8, 9)]:
            with self.subTest(stride=stride, dilation=dilation):
                torch.manual_seed(42)
                layer = CausalConv(2, 3, 9, stride=stride, dilation=dilation)
                source = torch.randn(2, 2, 257)
                expected = layer(source)
                state, pieces, offset = {}, [], 0
                for length in [1, 0, 3, 15, 2, 99, 7, 130]:
                    pieces.append(layer.step(source[..., offset:offset+length], state))
                    offset += length
                torch.testing.assert_close(torch.cat(pieces, dim=-1), expected)
                self.assertLess(state[id(layer)].shape[-1], dilation * 8 + 1)

    @torch.inference_mode()
    def test_transpose_overlap_and_bias_across_boundaries(self):
        for stride in (2, 4, 5, 8):
            with self.subTest(stride=stride):
                torch.manual_seed(4)
                layer = CausalTranspose(2, 3, stride)
                layer.bias.fill_(0.7)  # reveal a double-applied bias in overlap
                source = torch.randn(2, 2, 17)
                state, pieces, offset = {}, [], 0
                for length in [1, 0, 5, 2, 9]:
                    pieces.append(layer.step(source[..., offset:offset+length], state))
                    offset += length
                torch.testing.assert_close(torch.cat(pieces, dim=-1), layer(source))
                self.assertEqual(state[id(layer)].shape[-1], stride)


class TestStreamVCNeuralCore(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(6)
        self.model = StreamVCNeuralCore(2, 2, 2, 4).eval()
        self.source = torch.randn(1, 1, 320 * 6)
        self.side = torch.randn(1, 10, 6)
        with torch.inference_mode():
            self.speaker = self.model.encode_speaker(self.source)

    @torch.inference_mode()
    def test_framewise_matches_full_pass(self):
        expected = self.model(self.source, self.side, self.speaker)
        session = self.model.new_session()
        pieces = [session.push(self.source[..., i*320:(i+1)*320],
                               self.side[..., i:i+1], self.speaker) for i in range(6)]
        torch.testing.assert_close(torch.cat(pieces, dim=-1), expected, atol=2e-6, rtol=2e-5)
        self.assertEqual(expected.shape, self.source.shape)

    @torch.inference_mode()
    def test_bounded_state_reset_flush_and_session_isolation(self):
        first, second = self.model.new_session(), self.model.new_session()
        chunk, side = self.source[..., :320], self.side[..., :1]
        initial = first.push(chunk, side, self.speaker)
        size = first.state_bytes
        for _ in range(20):
            first.push(chunk, side, self.speaker)
            self.assertEqual(first.state_bytes, size)
        torch.testing.assert_close(second.push(chunk, side, self.speaker), initial)
        first.flush()
        self.assertEqual(first.state_bytes, 0)
        first.flush()
        with self.assertRaises(RuntimeError):
            first.push(chunk, side, self.speaker)
        first.reset()
        torch.testing.assert_close(first.push(chunk, side, self.speaker), initial)

    @torch.inference_mode()
    def test_no_future_leakage(self):
        changed = self.source.clone()
        changed[..., 320 * 3:] = 100
        a = self.model(self.source, self.side, self.speaker)
        b = self.model(changed, self.side, self.speaker)
        torch.testing.assert_close(a[..., :320*3], b[..., :320*3])

    def test_invalid_frame_or_feature_shape_does_not_advance_session(self):
        session = self.model.new_session()
        for source, side in [(self.source[..., :319], self.side[..., :1]),
                             (self.source[..., :320], self.side)]:
            with self.assertRaises(ValueError):
                session.push(source, side, self.speaker)
        self.assertEqual(session.state_bytes, 0)
