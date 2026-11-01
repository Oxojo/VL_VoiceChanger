import { StreamPlayer } from "./stream-audio.js";

const $ = (id) => document.getElementById(id);
const MAX_PENDING_SAMPLES = 32000; // two seconds, including the in-flight packet
let session = null;

function show(message, error = false) {
  $("liveStatus").textContent = message;
  $("liveStatus").classList.toggle("live-error", error);
}

function updateAvailability() {
  if (!session) $("liveStartBtn").disabled = !$("modelSelect").value;
}

function lockSettings(active) {
  document.body.dataset.streaming = String(active);
  const controls = document.querySelectorAll(
    "#modelSelect, #deviceSelect, #ortDevice, #outputSr, #pitch, #rmsMix, #denoise, #dehum, " +
    "#speaker, #seed, #hop, #context, #crossfade, #convertBtn, #recordBtn, input[name=mode]",
  );
  for (const el of controls) {
    if (active) { el.dataset.liveDisabled = String(el.disabled); el.disabled = true; }
    else if (el.dataset.liveDisabled !== undefined) {
      el.disabled = el.dataset.liveDisabled === "true";
      delete el.dataset.liveDisabled;
    }
  }
  $("liveStartBtn").disabled = active || !$("modelSelect").value;
  $("liveStopBtn").disabled = !active;
}

class LiveSession {
  constructor() {
    this.queue = [];
    this.pendingSamples = 0;
    this.inflight = 0;
    this.stopping = false;
    this.captureStopped = false;
    this.stopSent = false;
    this.closed = false;
    this.ready = false;
  }

  async start() {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia || !AudioContext) {
      throw new Error("マイクには localhost または HTTPS と対応ブラウザが必要です");
    }
    // Resume during the button gesture, before model loading or permission UI.
    this.context = new AudioContext();
    await this.context.resume();
    if (!this.context.audioWorklet) throw new Error("AudioWorklet 対応ブラウザを使ってください");
    await this.context.audioWorklet.addModule("/static/stream-capture.js");
    if (this.closed) return;
    const scheme = location.protocol === "https:" ? "wss:" : "ws:";
    this.ws = new WebSocket(`${scheme}//${location.host}/api/stream`);
    this.ws.binaryType = "arraybuffer";
    await new Promise((resolve, reject) => {
      this.rejectStart = reject;
      this.timeout = setTimeout(() => reject(new Error("モデルの準備がタイムアウトしました")), 120000);
      this.ws.onopen = () => {
        if (this.closed) return;
        this.ws.send(JSON.stringify({
          type: "start", model: $("modelSelect").value, device: $("deviceSelect").value,
          ort_cuda: $("ortDevice").value === "cuda",
          pitch: Number($("pitch").value), speaker: Number($("speaker").value),
          rms_mix: Number($("rmsMix").value), seed: Number($("seed").value || 1234),
          output_sr: Number($("outputSr").value), hop: Number($("hop").value),
          context: Number($("context").value), crossfade: Number($("crossfade").value),
          denoise: Number($("denoise").value), dehum: Number($("dehum").value),
        }));
      };
      this.ws.onmessage = (event) => {
        try {
          if (event.data instanceof ArrayBuffer) {
            if (!this.player || event.data.byteLength % 4) throw new Error("音声応答が不正です");
            this.player.enqueue(new Float32Array(event.data));
            return;
          }
          const message = JSON.parse(event.data);
          if (message.type === "ready") {
            this.ready = true;
            this.player = new StreamPlayer(this.context, message.output_sr);
            this.maxSamples = message.max_seconds * 16000;
            this.totalSamples = 0;
            $("liveBuffer").textContent = `入力蓄積 ${message.buffer_ms} ms ＋ 推論・通信・再生時間`;
            $("liveProviders").textContent = Object.entries(message.feature_providers || {})
              .map(([name, providers]) => `${name}: ${providers[0] === "CUDAExecutionProvider" ? "CUDA 優先" : "CPU"}`)
              .join(" / ");
            clearTimeout(this.timeout);
            resolve();
          } else if (message.type === "ack") {
            this.pendingSamples -= this.inflight;
            this.inflight = 0;
            this.stats(message);
            this.pump();
          } else if (message.type === "done") {
            this.done = true;
            this.stats(message);
            show("残りの音声を再生中…");
            clearTimeout(this.stopTimeout);
            this.endTimer = setTimeout(() => this.dispose("停止しました"),
              Math.ceil(this.player.queuedSeconds * 1000) + 100);
          } else if (message.type === "error") {
            throw new Error(message.message);
          }
        } catch (err) { reject(err); this.fail(err); }
      };
      this.ws.onerror = () => {
        const err = new Error("ストリーミングに接続できません。サーバーを確認してください");
        reject(err); this.fail(err);
      };
      this.ws.onclose = () => {
        if (this.closed || this.done) return;
        const err = new Error("接続が切れました。再開してください");
        reject(err); this.fail(err);
      };
    });
    if (this.closed) return;
    show("マイクの使用を許可してください…");
    const media = await navigator.mediaDevices.getUserMedia({ audio: {
      channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false,
    } });
    if (this.closed) { media.getTracks().forEach((t) => t.stop()); return; }
    this.media = media;
    for (const track of media.getTracks()) {
      track.onended = () => this.fail(new Error("マイクが切断されました"));
    }
    this.capture = new AudioWorkletNode(this.context, "vlvc-stream-capture", {
      numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1],
      channelCount: 1, channelCountMode: "explicit",
    });
    this.capture.onprocessorerror = () => this.fail(new Error("マイクの音声処理に失敗しました"));
    this.capture.port.onmessage = (event) => {
      if (this.closed) return;
      if (event.data.type === "audio") {
        const samples = event.data.samples;
        if (this.totalSamples + samples.length > this.maxSamples) { this.stop(); return; }
        this.totalSamples += samples.length;
        this.pendingSamples += samples.length;
        if (this.pendingSamples > MAX_PENDING_SAMPLES) {
          this.fail(new Error("変換が入力に追いつかないため停止しました。hop やデバイスを見直してください"));
          return;
        }
        this.queue.push(samples);
        this.pump();
      } else if (event.data.type === "stopped") {
        this.captureStopped = true;
        this.capture.disconnect();
        this.source.disconnect();
        this.pump();
      }
    };
    this.source = this.context.createMediaStreamSource(media);
    this.source.connect(this.capture);
    this.capture.connect(this.context.destination);
    this.context.onstatechange = () => {
      if (!this.closed && !this.done && this.context.state !== "running") {
        this.fail(new Error("音声再生が中断されました。画面を開いたまま再開してください"));
      }
    };
    show("変換中 — ヘッドホンでお試しください");
  }

  stats(message) {
    const names = { f0: "ピッチ推定", contentvec: "音声特徴", synth: "合成", post: "後処理" };
    $("liveStages").textContent = Object.entries(message.stage_ms || {}).length
      ? "区間あたり平均: " + Object.entries(message.stage_ms)
        .map(([name, ms]) => `${names[name] || name} ${ms.toFixed(1)} ms`).join(" / ")
      : "";
    $("liveMetrics").textContent =
      `入力 ${(message.input_frames / 100).toFixed(2)} 秒 / 出力 ${(message.emitted_frames / 100).toFixed(2)} 秒` +
      ` / 推論 RTF ${message.rtf === null ? "—" : message.rtf.toFixed(2)}` +
      ` / 最大推論 ${message.max_inference_ms.toFixed(0)} ms` +
      ` / 再生途切れ ${this.player.underruns} 回`;
    $("liveQueue").textContent =
      `送信待ち ${(this.pendingSamples / 16000).toFixed(2)} 秒 / 再生待ち ${this.player.queuedSeconds.toFixed(2)} 秒`;
  }

  pump() {
    if (this.closed || !this.ready || this.inflight || this.ws.readyState !== WebSocket.OPEN) return;
    const samples = this.queue.shift();
    if (samples) {
      this.inflight = samples.length;
      this.ws.send(samples.buffer);
    } else if (this.stopping && this.captureStopped && !this.stopSent) {
      this.stopSent = true;
      this.ws.send(JSON.stringify({ type: "stop" }));
    }
  }

  stop() {
    if (this.closed || this.stopping) return;
    if (!this.capture || this.done) { this.dispose("停止しました"); return; }
    this.stopping = true;
    this.stopTimeout = setTimeout(() => this.fail(new Error("停止処理がタイムアウトしました")), 30000);
    this.capture.port.postMessage({ type: "stop" });
    for (const track of this.media.getTracks()) { track.onended = null; track.stop(); }
    $("liveStopBtn").textContent = "すぐに切断";
    show("マイク停止 — 残りの音声を変換中…");
  }

  fail(err) { this.dispose(err.message, true); }

  dispose(message, error = false) {
    if (this.closed) return;
    this.closed = true;
    clearTimeout(this.timeout);
    clearTimeout(this.stopTimeout);
    clearTimeout(this.endTimer);
    this.rejectStart?.(new Error("停止しました"));
    if (this.capture) {
      this.capture.port.onmessage = null;
      this.capture.port.postMessage({ type: "stop" });
      this.capture.disconnect();
      this.capture.port.close();
    }
    this.source?.disconnect();
    for (const track of this.media?.getTracks() || []) { track.onended = null; track.stop(); }
    this.player?.stop();
    if (this.context) { this.context.onstatechange = null; this.context.close().catch(() => {}); }
    if (this.ws) { this.ws.onclose = null; this.ws.onerror = null; this.ws.onmessage = null; this.ws.close(); }
    this.queue = [];
    if (session === this) {
      session = null;
      lockSettings(false);
      $("liveStopBtn").textContent = "停止して残りを再生";
      show(message, error);
    }
  }
}

$("liveStartBtn").addEventListener("click", async () => {
  if (session) return;
  if (document.body.dataset.converting === "true" || $("recordBtn").classList.contains("on")) {
    show("ファイル変換・録音を停止してから開始してください", true);
    return;
  }
  if (!$("modelSelect").value) { show("先にモデルを選んでください", true); return; }
  if (Number($("denoise").value) || Number($("dehum").value)) {
    show("ノイズ除去とハム除去を 0 にしてから開始してください", true);
    return;
  }
  const current = new LiveSession();
  session = current;
  lockSettings(true);
  $("liveMetrics").textContent = "";
  $("liveQueue").textContent = "";
  $("liveProviders").textContent = "";
  $("liveStages").textContent = "";
  show("接続・モデル準備中…");
  try { await current.start(); } catch (err) { current.fail(err); }
});
$("liveStopBtn").addEventListener("click", () => {
  if (session?.stopping) session.dispose("切断しました");
  else session?.stop();
});
$("modelSelect").addEventListener("change", updateAvailability);
window.addEventListener("vlvc-models-ready", updateAvailability);
window.addEventListener("pagehide", () => session?.dispose("停止しました"));
updateAvailability();
