import { StreamPlayer } from "./stream-audio.js";
import { StreamDebug } from "./stream-debug.js";

const $ = (id) => document.getElementById(id);
const MAX_PENDING_SAMPLES = 32000; // two seconds, including the in-flight packet
let session = null;
let lastDebug = null;
let llvcInstalled = false;
let llvcInstalling = false;
const isLLVC = () => $("streamBackend").value === "llvc";
const canStart = () => isLLVC() ? llvcInstalled && !llvcInstalling : Boolean($("modelSelect").value);
const STREAM_PRESETS = {
  balanced: { hop: "0.5", context: "0.15", crossfade: "0.05" },
  quality: { hop: "0.3", context: "0.15", crossfade: "0.05" },
  low: { hop: "0.2", context: "0.06", crossfade: "0.02" },
};

function showPresetHint() {
  const name = $("streamPreset").value;
  $("streamPresetHint").textContent = name === "quality"
    ? "hop 300 / context 150 / crossfade 50 ms。先読みとクロスフェードは従来値を維持し、hopだけ縮めます。音質を比較して選んでください。"
    : name === "low"
    ? "hop 200 / context 60 / crossfade 20 ms。推論回数は毎秒5回になります。途切れや音質を確認し、追いつかない場合は従来設定へ戻してください。"
    : name === "balanced"
      ? "hop 500 / context 150 / crossfade 50 ms。現在の設定を基準に比較できます。"
      : "下の hop / context / crossfade を使用します。確定した入力蓄積は開始後に表示します。";
}

function renderDebug(debug, force = false) {
  const now = performance.now();
  if (!force && now - (debug.renderedAt || 0) < 250) return;
  debug.renderedAt = now;
  const latest = debug.records.at(-1);
  const packet = debug.lastPacket;
  const ms = (value) => Number.isFinite(value) ? `${value.toFixed(1)} ms` : "未取得";
  $("debugTotal").textContent = latest ? ms(latest.graph_capture_to_schedule_ms) : "出力待ち";
  const summary = debug.summary();
  const framing = debug.settings.framing;
  $("debugConfig").textContent = framing && debug.settings.backend === "llvc"
    ? `LLVC: 内部処理 ${framing.hop_ms} ms / 先読み ${framing.context_ms} ms / 入力送信20 ms単位。蓄積の実測は下表で確認してください。`
    : framing
    ? `設定値: hop ${framing.hop_ms} + context ${framing.context_ms} + crossfade ${framing.crossfade_ms} = ${framing.hop_ms + framing.context_ms + framing.crossfade_ms} ms（入力パケット20 msはこの蓄積に含む）`
    : "サーバーの設定値を待っています。";
  $("debugSummary").textContent = summary
    ? `直近 ${summary.count} 出力（全 ${summary.total_outputs}）: p50 ${ms(summary.p50_ms)} / p95 ${ms(summary.p95_ms)} / 最大 ${ms(summary.max_ms)}`
    : "推定値は音声出力ができてから表示します。";
  $("debugLiveState").textContent = debug.clientState
    ? `待ち音声量 ${ms(debug.clientState.pending_audio_ms)} / 応答待ち経過 ${ms(debug.clientState.inflight_elapsed_ms)}` : "";
  const rows = [
    ["入力蓄積", latest?.accumulation_ms, "出力先頭〜今回送信済み入力末尾（音声時間）"],
    ["入力末尾 → 送信", latest?.capture_end_to_send_ms, "推定。パケット受け渡しと送信待ちを含む"],
    ["ブラウザ内の送信待ち", latest?.input_queue_ms, "実測。同じ出力のパケット受け取り→送信。上段の内数"],
    ["送信 → 音声応答", latest?.audio_response_ms, "実測。サーバー処理と通信を含む"],
    ["サーバー内の合計", latest?.server_ms, "実測。受信→音声送信完了。次の処理時間を含む"],
    ["サーバーの実行待ち", latest?.executor_wait_ms, "実測。ワーカーへの投入→実行開始"],
    ["推論", latest?.inference_ms, "実測。下の処理段階を含む"],
    ["　LLVC特徴抽出・合成", latest?.stage_ms.llvc, "LLVC推論の内数"],
    ["　ピッチ推定", latest?.stage_ms.f0, "推論の内数"],
    ["　音声特徴抽出", latest?.stage_ms.contentvec, "推論の内数"],
    ["　合成", latest?.stage_ms.synth, "推論の内数"],
    ["　後処理", latest?.stage_ms.post, "推論の内数"],
    ["ワーカー内の推論以外", latest ? Math.max(0, latest.worker_ms - latest.inference_ms) : null, "実測差分。フレーム処理・出力結合など"],
    ["音声応答 → 再生開始予定", latest?.playback_schedule_ms, "音声グラフ上の待ち。再生待ちを含む"],
    ["直近パケットのACK往復", packet?.rtt_ms, "実測。音声の総遅延とは別"],
    ["ACK往復 − サーバー時間", packet?.transport_residual_ms, "参考。通信＋ブラウザ処理などの残差"],
    ["出力経路 baseLatency", debug.deviceLatencies?.base_latency_ms, "ブラウザ申告値。総推定値に加算しない"],
    ["出力デバイス outputLatency", debug.deviceLatencies?.output_latency_ms, "ブラウザ申告値。未対応の場合は未取得"],
  ];
  $("debugRows").innerHTML = rows.map(([label, value, note]) =>
    `<tr><th>${label}</th><td>${ms(value)}</td><td>${note}</td></tr>`).join("");
  $("debugExport").disabled = !(debug.lastPacket || debug.reason);
}

function show(message, error = false) {
  $("liveStatus").textContent = message;
  $("liveStatus").classList.toggle("live-error", error);
}

function updateAvailability() {
  if (!session) $("liveStartBtn").disabled = !canStart();
}

function lockSettings(active) {
  document.body.dataset.streaming = String(active);
  const controls = document.querySelectorAll(
    "#streamBackend, #llvcInstall, #modelSelect, #deviceSelect, #ortDevice, #featureCache, #streamPreset, #outputSr, #pitch, #rmsMix, #denoise, #dehum, " +
    "#speaker, #seed, #hop, #context, #crossfade, #liveDebug, #convertBtn, #recordBtn, input[name=mode]",
  );
  for (const el of controls) {
    if (active) { el.dataset.liveDisabled = String(el.disabled); el.disabled = true; }
    else if (el.dataset.liveDisabled !== undefined) {
      el.disabled = el.dataset.liveDisabled === "true";
      delete el.dataset.liveDisabled;
    }
  }
  $("liveStartBtn").disabled = active || !canStart();
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
    this.debug = $("liveDebug").checked ? new StreamDebug({}) : null;
    this.requestSeq = 0;
    this.captureOrigin = null;
    this.lastStats = null;
  }

  async start() {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia || !AudioContext) {
      throw new Error("マイクには localhost または HTTPS と対応ブラウザが必要です");
    }
    // Resume during the button gesture, before model loading or permission UI.
    this.context = new AudioContext({ latencyHint: "interactive" });
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
        const settings = isLLVC() ? {
          type: "start", backend: "llvc", model: "llvc-default",
          device: $("deviceSelect").value, debug: Boolean(this.debug),
        } : {
          type: "start", model: $("modelSelect").value, device: $("deviceSelect").value,
          ort_cuda: $("ortDevice").value === "cuda",
          feature_cache: $("featureCache").checked,
          debug: Boolean(this.debug),
          pitch: Number($("pitch").value), speaker: Number($("speaker").value),
          rms_mix: Number($("rmsMix").value), seed: Number($("seed").value || 1234),
          output_sr: Number($("outputSr").value), hop: Number($("hop").value),
          context: Number($("context").value), crossfade: Number($("crossfade").value),
          denoise: Number($("denoise").value), dehum: Number($("dehum").value),
        };
        if (this.debug) this.debug.settings = { ...settings, client_preset: $("streamPreset").value };
        this.ws.send(JSON.stringify(settings));
      };
      this.ws.onmessage = (event) => {
        try {
          if (event.data instanceof ArrayBuffer) {
            if (!this.player || event.data.byteLength % 4) throw new Error("音声応答が不正です");
            const receivedMs = this.debug ? performance.now() : 0;
            const scheduled = this.player.enqueue(new Float32Array(event.data));
            if (this.debug) this.debugOutput = { ...scheduled, receivedMs };
            return;
          }
          const message = JSON.parse(event.data);
          if (message.type === "ready") {
            this.ready = true;
            this.player = new StreamPlayer(this.context, message.output_sr, 3, (message.playback_lead_ms ?? 50) / 1000);
            this.maxSamples = message.max_seconds * 16000;
            this.totalSamples = 0;
            $("liveWarmup").textContent = message.warmup_ms > 0
              ? `初回推論の事前準備 ${message.warmup_ms.toFixed(0)} ms（マイク開始前・遅延統計には含めません）`
              : "";
            if (this.debug) {
              if (!message.debug) throw new Error("サーバーが debug モードに対応していません");
              this.debug.settings = { ...this.debug.settings, framing: message.debug,
                output_sr: message.output_sr, warmup_ms: message.warmup_ms,
                feature_cache_config: message.feature_cache, runtime: message.runtime,
                runtime_validation_max_error: message.runtime_validation_max_error };
              this.debug.device(this.context);
              renderDebug(this.debug, true);
            }
            $("liveBuffer").textContent = `入力蓄積 ${message.buffer_ms} ms ＋ 推論・通信・再生時間`;
            $("liveProviders").textContent = message.backend === "llvc" ? `LLVC: ${message.runtime || "実行方式未取得"}` : Object.entries(message.feature_providers || {})
              .map(([name, providers]) => `${name}: ${providers[0] === "CUDAExecutionProvider" ? "CUDA 優先" : "CPU"}`)
              .join(" / ");
            clearTimeout(this.timeout);
            resolve();
          } else if (message.type === "ack") {
            this.recordDebug(message);
            this.pendingSamples -= this.inflight;
            this.inflight = 0;
            this.stats(message);
            this.pump();
            this.updateDebugState();
          } else if (message.type === "done") {
            this.recordDebug(message);
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
      processorOptions: { debug: Boolean(this.debug) },
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
          const rtf = this.lastStats?.rtf;
          const detail = Number.isFinite(rtf) ? `（推論RTF ${rtf.toFixed(2)}、送信待ち ${(this.pendingSamples / 16).toFixed(0)} ms）` : "";
          this.fail(new Error(isLLVC()
            ? `LLVCの処理が入力に追いつかないため停止しました${detail}。debugの診断JSONで推論時間と送信待ちを確認してください`
            : "変換が入力に追いつかないため停止しました。hop やデバイスを見直してください"));
          return;
        }
        if (this.debug) {
          if (this.captureOrigin === null) this.captureOrigin = event.data.captureStart;
          this.queue.push({ samples, captureEnd: event.data.captureEnd, receivedMs: performance.now() });
        } else this.queue.push(samples);
        this.pump();
        this.updateDebugState();
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

  recordDebug(message) {
    if (!this.debug) return;
    this.debug.record({ packet: this.debugPacket, server: message.debug,
      output: this.debugOutput, ackMs: performance.now(), captureOrigin: this.captureOrigin,
      outputSr: this.player.sampleRate });
    this.debugRequestPending = false;
    this.debugOutput = null;
    this.debug.device(this.context);
    renderDebug(this.debug, message.type === "done");
  }

  markDebugSend(packet) {
    if (!this.debug) return;
    this.debugRequestPending = true;
    this.debugPacket = { receivedMs: packet.receivedMs, captureEnd: packet.captureEnd, seq: ++this.requestSeq,
      sentMs: performance.now(), sentGraphTime: this.context.currentTime };
  }

  updateDebugState() {
    if (!this.debug) return;
    this.debug.clientState = {
      pending_audio_ms: this.pendingSamples / 16,
      inflight_elapsed_ms: this.debugRequestPending ? performance.now() - this.debugPacket.sentMs : null,
      inflight_request_seq: this.debugRequestPending ? this.debugPacket.seq : null,
      state: this.closed ? "closed" : this.stopping ? "stopping" : "running",
    };
    renderDebug(this.debug);
  }

  stats(message) {
    this.lastStats = message;
    const cache = message.feature_cache;
    const avoided = cache?.enabled && cache.requested_frames
      ? cache.avoided_frames * 100 / cache.requested_frames : 0;
    $("liveCache").textContent = cache?.enabled
      ? `特徴量キャッシュ試験: 抽出入力を ${avoided.toFixed(1)}% 省略 / 再利用 ${cache.reused_frames} フレーム / 保持 ${cache.retained_frames} フレーム（境界${cache.guard_ms} ms）`
      : "";
    if (this.debug) this.debug.latestFeatureCache = cache;
    const names = { llvc: "LLVC推論", f0: "ピッチ推定", contentvec: "音声特徴", synth: "合成", post: "後処理" };
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
    const packet = this.queue.shift();
    if (packet) {
      const samples = this.debug ? packet.samples : packet;
      this.inflight = samples.length;
      this.markDebugSend(packet);
      this.ws.send(samples.buffer);
    } else if (this.stopping && this.captureStopped && !this.stopSent) {
      this.stopSent = true;
      this.markDebugSend({ receivedMs: performance.now(),
        captureEnd: this.captureOrigin + this.totalSamples / 16000 });
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
    if (this.debug) {
      this.debug.reason = message;
      this.updateDebugState();
      renderDebug(this.debug, true);
    }
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
  if (!canStart()) { show(isLLVC() ? "先にLLVCモデルを取得してください" : "先にモデルを選んでください", true); return; }
  if (!isLLVC() && (Number($("denoise").value) || Number($("dehum").value))) {
    show("ノイズ除去とハム除去を 0 にしてから開始してください", true);
    return;
  }
  const current = new LiveSession();
  session = current;
  lastDebug = current.debug;
  $("debugExport").disabled = true;
  if (lastDebug) renderDebug(lastDebug, true);
  lockSettings(true);
  $("liveMetrics").textContent = "";
  $("liveQueue").textContent = "";
  $("liveProviders").textContent = "";
  $("liveStages").textContent = "";
  $("liveCache").textContent = "";
  $("liveWarmup").textContent = "";
  show("接続・モデル読み込み・初回推論の準備中…（準備が終わってからマイクを開始します）");
  try { await current.start(); } catch (err) { current.fail(err); }
});
$("liveStopBtn").addEventListener("click", () => {
  if (session?.stopping) session.dispose("切断しました");
  else session?.stop();
});
$("modelSelect").addEventListener("change", updateAvailability);
$("streamPreset").addEventListener("change", () => {
  const preset = STREAM_PRESETS[$("streamPreset").value];
  if (preset) {
    for (const [name, value] of Object.entries(preset)) $(name).value = value;
    const mode = document.querySelector('input[name="mode"][value="chunk"]');
    mode.checked = true;
    mode.dispatchEvent(new Event("change"));
  }
  showPresetHint();
});
for (const name of ["hop", "context", "crossfade"]) {
  $(name).addEventListener("input", () => {
    $("streamPreset").value = "custom";
    showPresetHint();
  });
}
showPresetHint();
window.addEventListener("vlvc-models-ready", updateAvailability);
window.addEventListener("pagehide", () => session?.dispose("停止しました"));
$("liveDebug").addEventListener("change", () => $("debugPanel").classList.toggle("hidden", !$("liveDebug").checked));
$("debugExport").addEventListener("click", () => {
  if (!lastDebug || !(lastDebug.lastPacket || lastDebug.reason)) return;
  const blob = new Blob([JSON.stringify(lastDebug.snapshot(), null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "vlvc-stream-debug.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
updateAvailability();

async function refreshLLVC() {
  try {
    const response = await fetch('/api/llvc');
    if (!response.ok) throw new Error('モデル状態を確認できません');
    const data = await response.json();
    llvcInstalled = data.installed;
    $("llvcStatus").textContent = llvcInstalled
      ? 'モデル配置済み。「マイク変換を開始」で試せます。100ms以下の達成は未検証です。'
      : 'LLVCモデルが必要です。「LLVCモデルを取得」を押してください。';
  } catch (error) { $("llvcStatus").textContent = error.message; }
  updateAvailability();
}
$("streamBackend").addEventListener('change', () => {
  $("llvcSetup").hidden = !isLLVC();
  $("streamPreset").disabled = isLLVC();
  $("streamPresetHint").textContent = isLLVC()
    ? 'LLVCは内部13ms単位・入力パケット20ms。下のRVC専用設定は使いません。デバイスとdebugモードは共通です。'
    : '';
  if (!isLLVC()) showPresetHint();
  if (isLLVC()) { $("deviceSelect").value = 'cpu'; refreshLLVC(); }
  $("liveBuffer").textContent = isLLVC() ? '入力蓄積・実測処理時間は開始後に表示します。' : '設定されたRVCの入力蓄積に加え、推論・通信・再生に時間がかかります。';
  updateAvailability();
});
$("llvcInstall").addEventListener('click', async () => {
  if (session || llvcInstalling) return;
  llvcInstalling = true;
  $("llvcInstall").disabled = true;
  $("llvcStatus").textContent = 'LLVCの公開モデルを取得中…（初回のみ・数分かかる場合があります）';
  updateAvailability();
  try {
    const response = await fetch('/api/llvc/install', {method: 'POST'});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'モデル取得に失敗しました');
    await refreshLLVC();
  } catch (error) { $("llvcStatus").textContent = error.message; }
  finally { llvcInstalling = false; $("llvcInstall").disabled = Boolean(session); updateAvailability(); }
});
