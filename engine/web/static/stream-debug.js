// Browser monotonic durations and server monotonic durations are kept separate.
// Never subtract raw timestamps from different machines/clocks.
const positive = (value) => Math.max(0, value);
const reportedLatency = (seconds) =>
  Number.isFinite(seconds) && seconds >= 0 ? seconds * 1000 : null;

export class StreamDebug {
  constructor(settings, maxRecords = 300) {
    this.settings = settings;
    this.maxRecords = maxRecords;
    this.records = [];
    this.lastPacket = null;
    this.totalOutputs = 0;
    this.reason = null;
  }

  record({ packet, server, output, ackMs, captureOrigin, outputSr }) {
    if (!server || server.request_seq !== packet.seq) throw new Error("debug 応答の順序が一致しません");
    const rtt = positive(ackMs - packet.sentMs);
    this.lastPacket = {
      request_seq: packet.seq,
      input_queue_ms: positive(packet.sentMs - packet.receivedMs),
      rtt_ms: rtt,
      server_ms: server.server_ms,
      executor_wait_ms: server.executor_wait_ms,
      worker_ms: server.worker_ms,
      inference_ms: server.inference_ms,
      transport_residual_ms: positive(rtt - server.server_ms),
    };
    if (!output) {
      if (server.output_frames !== 0) throw new Error("debug 音声応答がありません");
      return;
    }
    if (Math.abs(output.firstSample / outputSr - server.output_start_frame / 100) > 1e-6
        || Math.abs(output.samples - server.output_frames * (outputSr / 100)) > 1e-6) {
      throw new Error("debug 音声フレーム位置が一致しません");
    }
    const sourceTime = captureOrigin + server.output_start_frame / 100;
    const record = {
      ...this.lastPacket,
      output_start_frame: server.output_start_frame,
      output_frames: server.output_frames,
      graph_capture_to_schedule_ms: positive((output.startTime - sourceTime) * 1000),
      accumulation_ms: positive((packet.captureEnd - sourceTime) * 1000),
      capture_end_to_send_ms: positive((packet.sentGraphTime - packet.captureEnd) * 1000),
      audio_response_ms: positive(output.receivedMs - packet.sentMs),
      playback_schedule_ms: positive((output.startTime - output.graphTime) * 1000),
      stage_ms: { ...server.stage_ms },
      feature_cache: server.feature_cache || { enabled: false },
    };
    this.records.push(record);
    if (this.records.length > this.maxRecords) this.records.shift();
    this.totalOutputs += 1;
  }

  summary() {
    if (!this.records.length) return null;
    const values = this.records.map((r) => r.graph_capture_to_schedule_ms).sort((a, b) => a - b);
    const percentile = (p) => values[Math.max(0, Math.ceil(values.length * p) - 1)];
    return { count: values.length, total_outputs: this.totalOutputs,
      p50_ms: percentile(0.5), p95_ms: percentile(0.95), max_ms: values.at(-1) };
  }

  device(context) {
    this.deviceLatencies = {
      base_latency_ms: reportedLatency(context.baseLatency),
      output_latency_ms: reportedLatency(context.outputLatency),
    };
  }

  snapshot() {
    return {
      version: 1, settings: this.settings, device_estimates: this.deviceLatencies,
      summary: this.summary(), last_packet: this.lastPacket, reason: this.reason,
      client_state: this.clientState,
      feature_cache: this.latestFeatureCache,
      limitations: [
        "Graph capture to scheduled playback is an estimate, not physical microphone-to-ear latency.",
        "Input device/OS capture latency is not measured. Browser-reported output latencies are estimates.",
        "RTT minus server time includes transport and browser dispatch; it is not one-way network latency.",
        "Rows overlap and use different clocks; do not add all rows. Stage timings are inside inference.",
        `Only the latest ${this.maxRecords} output records are retained. No audio samples are exported.`,
      ],
      records: this.records,
    };
  }
}
