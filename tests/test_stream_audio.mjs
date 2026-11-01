import assert from "node:assert/strict";
import { test } from "node:test";
import { CaptureResampler, StreamPlayer } from "../engine/web/static/stream-audio.js";

function resample(input, sr, sizes) {
  const packets = [];
  const converter = new CaptureResampler(sr, (packet) => packets.push(packet));
  let offset = 0;
  let index = 0;
  while (offset < input.length) {
    const n = sizes[index++ % sizes.length];
    converter.push(input.subarray(offset, offset + n));
    offset += n;
  }
  converter.flush();
  return packets;
}

test("capture phase and packet boundaries survive arbitrary callback splits", () => {
  for (const sr of [16000, 44100, 48000]) {
    const input = Float32Array.from({ length: sr + 113 }, (_, i) => Math.sin(i / 17) * 0.3);
    const baseline = resample(input, sr, [input.length]);
    for (const sizes of [[1], [7], [128], [17, 4096, 3, 128]]) {
      const packets = resample(input, sr, sizes);
      assert.deepEqual(packets, baseline);
      assert.equal(packets.reduce((n, p) => n + p.length, 0), Math.floor(input.length * 16000 / sr));
      assert(packets.slice(0, -1).every((p) => p.length === 320));
    }
  }
});

test("capture averages input samples and preserves DC gain", () => {
  const packets = resample(Float32Array.from({ length: 960 }, (_, i) => i), 48000, [128]);
  assert.equal(packets.length, 1);
  for (let i = 0; i < 320; i++) assert.equal(packets[0][i], 3 * i + 1);
  for (const sr of [44100, 48000]) {
    const dc = resample(new Float32Array(sr).fill(0.25), sr, [128]);
    assert.equal(dc.length, 50);
    assert(dc.every((p) => p.every((x) => x === 0.25)));
  }
});

test("flush emits a partial packet only once without extra samples", () => {
  const packets = [];
  const converter = new CaptureResampler(48000, (p) => packets.push(p));
  converter.push(new Float32Array(301));
  converter.flush();
  converter.flush();
  assert.equal(packets.length, 1);
  assert.equal(packets[0].length, 100);
});

test("transferring a packet does not detach subsequent capture buffers", () => {
  const packets = [];
  const converter = new CaptureResampler(48000, (packet) => {
    packets.push(structuredClone(packet, { transfer: [packet.buffer] }));
  });
  converter.push(new Float32Array(48000).fill(0.25));
  converter.flush();
  assert.equal(packets.length, 50);
  assert(packets.every((p) => p.length === 320 && p.every((x) => x === 0.25)));
});

function fakeContext() {
  const context = {
    currentTime: 1, destination: {}, buffers: [], sources: [],
    createBuffer(channels, length, sr) {
      const buffer = { channels, length, sr, copyToChannel(samples) { this.data = samples.slice(); } };
      this.buffers.push(buffer);
      return buffer;
    },
    createBufferSource() {
      const source = {
        connect() {}, disconnect() { this.disconnected = true; },
        start(time) { this.startTime = time; }, stop() { this.stopped = true; },
      };
      this.sources.push(source);
      return source;
    },
  };
  return context;
}

test("playback schedules contiguous audio at the negotiated sample rate", () => {
  const context = fakeContext();
  const player = new StreamPlayer(context, 48000);
  player.enqueue(new Float32Array(9600).fill(0.2));
  context.currentTime = 1.24; // still buffered: do not insert another 50 ms gap
  player.enqueue(new Float32Array(4800).fill(0.4));
  assert.equal(context.sources[0].startTime, 1.05);
  assert.equal(context.sources[1].startTime, 1.25);
  assert.equal(player.samples, 14400);
  assert.equal(player.underruns, 0);
  assert.deepEqual(context.buffers.map((b) => b.sr), [48000, 48000]);
  assert.equal(context.buffers[0].data[0], Math.fround(0.2));
  context.sources[0].onended();
  assert.equal(player.sources.size, 1);
  player.stop();
  assert.equal(player.sources.size, 0);
  assert(context.sources[1].stopped);
  assert.equal(player.queuedSeconds, 0);
});

test("playback counts underruns and rejects excessive queued latency", () => {
  const context = fakeContext();
  const player = new StreamPlayer(context, 16000, 1);
  player.enqueue(new Float32Array(8000));
  context.currentTime = 2;
  player.enqueue(new Float32Array(8000));
  assert.equal(player.underruns, 1);
  assert.equal(context.sources[1].startTime, 2.05);
  assert.throws(() => player.enqueue(new Float32Array(16000)), /再生待ち/);
  assert.equal(player.sources.size, 2);
  player.stop();
});
