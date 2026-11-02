import { CaptureResampler } from "./stream-audio.js";

class StreamCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.active = true;
    this.debug = Boolean(options.processorOptions?.debug);
    this.captureOrigin = null;
    this.sentSamples = 0;
    this.resampler = new CaptureResampler(sampleRate, (samples) => {
      const message = { type: "audio", samples };
      if (this.debug) {
        message.captureStart = this.captureOrigin + this.sentSamples / 16000;
        this.sentSamples += samples.length;
        message.captureEnd = this.captureOrigin + this.sentSamples / 16000;
      }
      this.port.postMessage(message, [samples.buffer]);
    });
    this.port.onmessage = (event) => {
      if (event.data.type === "stop" && this.active) {
        this.active = false;
        this.resampler.flush();
        this.port.postMessage({ type: "stopped" });
      }
    };
  }

  process(inputs) {
    if (this.active && inputs[0]?.[0]) {
      if (this.debug && this.captureOrigin === null) this.captureOrigin = currentTime;
      this.resampler.push(inputs[0][0]);
    }
    // Output remains silent: never monitor raw microphone audio.
    return this.active;
  }
}

registerProcessor("vlvc-stream-capture", StreamCapture);
