import assert from "node:assert/strict";
import { test } from "node:test";
import { StreamDebug } from "../engine/web/static/stream-debug.js";

function event(seq = 1) {
  return {
    packet: { seq, sentMs: 1010, receivedMs: 1000, sentGraphTime: 10.7, captureEnd: 10.7 },
    server: { request_seq: seq, server_ms: 80, executor_wait_ms: 5, worker_ms: 70,
      inference_ms: 60, output_start_frame: 0, output_frames: 50,
      stage_ms: { f0: 10, contentvec: 20, synth: 25, post: 3 } },
    output: { firstSample: 0, samples: 24000, receivedMs: 1110, graphTime: 10.8, startTime: 10.85 },
    ackMs: 1112, captureOrigin: 10, outputSr: 48000,
  };
}

test("debug separates graph estimates, measured durations and nested stage timings", () => {
  const debug = new StreamDebug({ hop: 0.5 });
  debug.record(event());
  const r = debug.records[0];
  assert(Math.abs(r.graph_capture_to_schedule_ms - 850) < 1e-9);
  assert(Math.abs(r.accumulation_ms - 700) < 1e-9);
  assert.equal(r.input_queue_ms, 10);
  assert.equal(r.audio_response_ms, 100);
  assert.equal(r.rtt_ms, 102);
  assert.equal(r.transport_residual_ms, 22);
  assert(Math.abs(r.playback_schedule_ms - 50) < 1e-9);
  assert.equal(r.inference_ms, 60);
  assert.equal(r.stage_ms.contentvec, 20);
});

test("output frame positions map later chunks to their corresponding input samples", () => {
  const input = event();
  input.server.output_start_frame = 50;
  input.output.firstSample = 24000;
  input.packet.captureEnd = 11.2;
  input.packet.sentGraphTime = 11.22;
  input.output.graphTime = 11.32;
  input.output.startTime = 11.37;
  const debug = new StreamDebug({});
  debug.record(input);
  assert(Math.abs(debug.records[0].graph_capture_to_schedule_ms - 870) < 1e-9);
  assert(Math.abs(debug.records[0].accumulation_ms - 700) < 1e-9);
  assert(Math.abs(debug.records[0].capture_end_to_send_ms - 20) < 1e-9);
});

test("short flush uses actual captured duration instead of the configured 700 ms", () => {
  const input = event();
  input.packet.captureEnd = 10.05;
  input.packet.sentGraphTime = 10.06;
  input.server.output_frames = 5;
  input.output.samples = 2400;
  const debug = new StreamDebug({ hop_ms: 500, context_ms: 150, crossfade_ms: 50 });
  debug.record(input);
  assert(Math.abs(debug.records[0].accumulation_ms - 50) < 1e-9);
});

test("ACKs without output update packet measurements without inventing audio latency", () => {
  const input = event();
  input.output = null;
  input.server.output_frames = 0;
  const debug = new StreamDebug({});
  debug.record(input);
  assert.equal(debug.lastPacket.rtt_ms, 102);
  assert.equal(debug.records.length, 0);
  assert.equal(debug.summary(), null);
});

test("rolling summaries and exports are bounded and contain no audio", () => {
  const debug = new StreamDebug({ debug: true }, 3);
  for (let i = 1; i <= 5; i++) {
    const input = event(i);
    input.output.startTime = 10 + i / 10;
    debug.record(input);
  }
  assert.equal(debug.records.length, 3);
  assert.equal(debug.summary().total_outputs, 5);
  assert(Math.abs(debug.summary().p50_ms - 400) < 1e-9);
  assert(Math.abs(debug.summary().p95_ms - 500) < 1e-9);
  debug.device({ baseLatency: 0.01 });
  debug.reason = "stopped";
  const exported = JSON.parse(JSON.stringify(debug.snapshot()));
  assert.equal(exported.device_estimates.base_latency_ms, 10);
  assert.equal(exported.device_estimates.output_latency_ms, null);
  assert.equal(exported.records.length, 3);
  assert(exported.limitations.some((x) => x.includes("not physical")));
  assert(!JSON.stringify(exported).includes('"samples"'));
});

test("sequence and frame mismatches cannot produce misleading debug results", () => {
  const debug = new StreamDebug({});
  const wrongSequence = event();
  wrongSequence.server.request_seq = 2;
  assert.throws(() => debug.record(wrongSequence), /順序/);
  const wrongFrame = event();
  wrongFrame.output.firstSample = 480;
  assert.throws(() => debug.record(wrongFrame), /フレーム位置/);
  assert.equal(debug.records.length, 0);
});

test("LLVC 13ms outputs tolerate floating point 10ms-frame fractions", () => {
  const input = event();
  input.outputSr = 16000;
  input.output.firstSample = 416;
  input.output.samples = 208;
  input.server.output_start_frame = 2.6;
  input.server.output_frames = 3.9 - 2.6;
  const debug = new StreamDebug({ backend: 'llvc' });
  debug.record(input);
  assert.equal(debug.records.length, 1);
});
