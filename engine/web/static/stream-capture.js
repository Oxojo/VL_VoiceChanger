import { CaptureResampler } from "./stream-audio.js";

class StreamCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.active = true;
    this.resampler = new CaptureResampler(sampleRate, (samples) => {
      this.port.postMessage({ type: "audio", samples }, [samples.buffer]);
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
    if (this.active && inputs[0]?.[0]) this.resampler.push(inputs[0][0]);
    // Output remains silent: never monitor raw microphone audio.
    return this.active;
  }
}

registerProcessor("vlvc-stream-capture", StreamCapture);
