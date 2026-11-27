// Stateful box-filter resampling: phase and partial packets survive callbacks.
// Keeping this independent of browser globals also permits deterministic tests.
export class CaptureResampler {
  constructor(inputRate, onPacket, packetSamples = 320) {
    this.ratio = inputRate / 16000;
    this.remaining = this.ratio;
    this.sum = 0;
    this.packet = new Float32Array(packetSamples);
    this.offset = 0;
    this.onPacket = onPacket;
  }

  push(samples) {
    for (const sample of samples) {
      let available = 1;
      while (available > 1e-9) {
        const weight = Math.min(available, this.remaining);
        this.sum += sample * weight;
        available -= weight;
        this.remaining -= weight;
        if (this.remaining < 1e-9) {
          this.packet[this.offset++] = this.sum / this.ratio;
          this.sum = 0;
          this.remaining = this.ratio;
          if (this.offset === this.packet.length) {
            const completed = this.packet;
            this.packet = new Float32Array(completed.length);
            this.offset = 0;
            // postMessage may transfer/detach completed.buffer immediately.
            this.onPacket(completed);
          }
        }
      }
    }
  }

  flush() {
    if (this.offset) this.onPacket(this.packet.slice(0, this.offset));
    this.offset = 0;
  }
}

export class StreamPlayer {
  constructor(context, sampleRate, maxSeconds = 3) {
    this.context = context;
    this.sampleRate = sampleRate;
    this.maxSeconds = maxSeconds;
    this.nextTime = 0;
    this.sources = new Set();
    this.underruns = 0;
    this.samples = 0;
  }

  get queuedSeconds() {
    return Math.max(0, this.nextTime - this.context.currentTime);
  }

  enqueue(samples) {
    if (!samples.length) return;
    if (this.queuedSeconds + samples.length / this.sampleRate > this.maxSeconds) {
      throw new Error("再生待ちが長すぎます。設定を見直して再開してください");
    }
    const now = this.context.currentTime;
    if (this.samples && this.nextTime < now) this.underruns += 1;
    const start = this.samples && this.nextTime >= now ? this.nextTime : now + 0.05;
    const buffer = this.context.createBuffer(1, samples.length, this.sampleRate);
    buffer.copyToChannel(samples, 0);
    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.context.destination);
    this.sources.add(source);
    source.onended = () => { this.sources.delete(source); source.disconnect(); };
    source.start(start);
    this.nextTime = start + samples.length / this.sampleRate;
    const firstSample = this.samples;
    this.samples += samples.length;
    return { startTime: start, graphTime: now, firstSample, samples: samples.length };
  }

  stop() {
    for (const source of this.sources) {
      source.onended = null;
      source.stop();
      source.disconnect();
    }
    this.sources.clear();
    this.nextTime = 0;
  }
}
