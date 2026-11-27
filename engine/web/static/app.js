"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  wavBlob: null,
  uploadName: "input.wav",
  inputUrl: null,
  name: "",
  buffer: null,
  original: null,
  cleaned: false,
  jobId: null,
  poll: null,
  startedAt: 0,
  recorder: null,
  stream: null,
  recTimer: null,
  recSeconds: 0,
  models: [],
};

// ------------------------------------------------------------------ 共通
async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch (_) { /* JSON でない応答はそのまま */ }
    throw new Error(detail);
  }
  return res.json();
}

function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.classList.add("show");
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => el.classList.remove("show"), 5000);
}

const fmt = (value, digits = 2) =>
  value === null || value === undefined || Number.isNaN(value) ? "—" : Number(value).toFixed(digits);

function clock(seconds) {
  const m = Math.floor(seconds / 60).toString().padStart(2, "0");
  const s = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

// ------------------------------------------------------------------ 波形
function encodeWav(samples, sampleRate, channels = [samples]) {
  const channelCount = channels.length;
  const dataBytes = samples.length * channelCount * 2;
  const buffer = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buffer);
  const writeString = (offset, text) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };
  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channelCount, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channelCount * 2, true);
  view.setUint16(32, channelCount * 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, dataBytes, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    for (const channel of channels) {
      const s = Math.max(-1, Math.min(1, channel[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }
  }
  return new Blob([view], { type: "audio/wav" });
}

function drawWave(canvas, samples) {
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 600;
  const height = canvas.clientHeight || 72;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);

  ctx.strokeStyle = "#26324e";
  ctx.beginPath();
  ctx.moveTo(0, height / 2);
  ctx.lineTo(width, height / 2);
  ctx.stroke();

  if (!samples || !samples.length) return;
  const step = Math.max(1, Math.floor(samples.length / width));
  ctx.strokeStyle = "#4cc9f0";
  ctx.beginPath();
  for (let x = 0; x < width; x += 1) {
    let min = 1;
    let max = -1;
    const start = x * step;
    for (let i = 0; i < step && start + i < samples.length; i += 1) {
      const v = samples[start + i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    ctx.moveTo(x + 0.5, (1 - max) * height / 2);
    ctx.lineTo(x + 0.5, (1 - min) * height / 2);
  }
  ctx.stroke();
}

// ------------------------------------------------------------------ 入力
async function decodeToWav(file) {
  const arrayBuffer = await file.arrayBuffer();
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  let decoded;
  try {
    decoded = await ctx.decodeAudioData(arrayBuffer);
  } finally {
    await ctx.close();
  }

  // 対応形式は再エンコードしない。波形表示用のデコード結果は試聴・送信に使わない。
  // 16 kHz 化とモノラル化は、サーバーのエンジン内部でのみ行う。
  const samples = decoded.getChannelData(0);
  const suffix = /\.(wav|flac|ogg|mp3)$/i.exec(file.name || "")?.[1].toLowerCase();
  const mimeSuffix = {
    "audio/wav": "wav", "audio/x-wav": "wav", "audio/wave": "wav",
    "audio/flac": "flac", "audio/ogg": "ogg", "audio/mpeg": "mp3",
  }[file.type];
  const format = suffix || mimeSuffix;
  if (format) {
    return {
      blob: file, samples, duration: decoded.duration,
      uploadName: `input.${format}`, description: "元ファイルをそのまま試聴・送信（加工なし）",
    };
  }

  // M4A などサーバーで読めない形式のみ、ブラウザのデコード周波数の WAV にする。
  return {
    blob: encodeWavBuffer(decoded), samples, duration: decoded.duration,
    uploadName: "input.wav", description: `${decoded.sampleRate} Hz WAV に変換（16 kHz 化なし）`,
  };
}

function encodeWavBuffer(decoded) {
  // 録音はモノラル。複数チャネルの場合もここでは勝手にダウンミックスしない。
  const channels = Array.from({ length: decoded.numberOfChannels }, (_, i) => decoded.getChannelData(i));
  return encodeWav(channels[0], decoded.sampleRate, channels);
}

function renderInput(blob, samples, name, info, uploadName = "input.wav") {
  state.buffer = samples;
  state.wavBlob = blob;
  state.name = name;
  state.uploadName = uploadName;

  $("inputInfo").textContent = info;
  drawWave($("inputWave"), samples);
  const player = $("inputAudio");
  player.pause();
  if (state.inputUrl) URL.revokeObjectURL(state.inputUrl);
  state.inputUrl = URL.createObjectURL(blob);
  player.src = state.inputUrl;
  player.classList.remove("hidden");
  $("convertBtn").disabled = document.body.dataset.streaming === "true";
  if (document.body.dataset.streaming === "true") $("convertBtn").dataset.liveDisabled = "false";
  $("cleanBtn").disabled = false;
  $("convertHint").textContent = "変換できます";
}

function adoptInput(blob, samples, name, info, uploadName = "input.wav") {
  // 新しく音声を読み込んだときは、元に戻せるように控えておく
  state.original = { blob, samples, name, uploadName };
  state.cleaned = false;
  $("revertBtn").classList.add("hidden");
  $("cleanInfo").textContent = "録音・読み込みのあと、この音声だけに除去をかけられます（任意）";
  renderInput(blob, samples, name, info, uploadName);
}

async function setInputFromBuffer(decoded, name) {
  const samples = decoded.getChannelData(0);

  adoptInput(
    encodeWavBuffer(decoded),
    samples,
    name,
    `${name} ／ ${decoded.duration.toFixed(2)} 秒 ／ ${decoded.sampleRate} Hz WAV（内部解析のみ 16 kHz）`,
  );
}

async function setInputFromFile(file) {
  if (!file) return;
  if (!file.type.startsWith("audio/") && !/\.(wav|mp3|flac|ogg|m4a)$/i.test(file.name)) {
    toast("音声ファイルを選んでください");
    return;
  }
  try {
    const { blob, samples, duration, uploadName, description } = await decodeToWav(file);
    adoptInput(blob, samples, file.name, `${file.name} ／ ${duration.toFixed(2)} 秒 ／ ${description}`, uploadName);
  } catch (err) {
    toast(`音声を読めませんでした: ${err.message}`);
  }
}

// ------------------------------------------------------------------ 入力のノイズ除去（任意）
async function applyDenoiseToInput() {
  if (!state.wavBlob) return;
  const strength = Number($("denoise").value);
  const dehum = Number($("dehum").value);
  if (strength === 0 && dehum === 0) {
    toast("先に「ノイズ除去」の強さ、または「ハム除去」を選んでください");
    return;
  }

  $("cleanBtn").disabled = true;
  $("cleanInfo").textContent = "ノイズ除去中…";
  try {
    const form = new FormData();
    form.append("audio", state.wavBlob, state.uploadName);
    form.append("strength", String(strength));
    form.append("dehum", String(dehum));

    const res = await api("/api/denoise", { method: "POST", body: form });
    const blob = await (await fetch(`${res.audio}?t=${Date.now()}`)).blob();
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const decoded = await ctx.decodeAudioData(await blob.arrayBuffer());
    await ctx.close();

    renderInput(
      blob,
      decoded.getChannelData(0),
      `${state.name.replace(/（除去済み）$/, "")}（除去済み）`,
      `ノイズ除去を適用: ${fmt(res.before_db, 1)} → ${fmt(res.after_db, 1)} dBFS ／ ${fmt(res.duration, 2)} 秒`,
    );
    state.cleaned = true;
    $("revertBtn").classList.remove("hidden");
    $("cleanInfo").textContent = `ノイズ除去済み（${fmt(res.before_db, 1)} → ${fmt(res.after_db, 1)} dBFS）`;

    // 二重がけを避けるため、変換時の除去は 0 に戻す
    if (strength > 0) {
      $("denoise").value = "0";
      $("denoiseValue").textContent = "0.00";
    }
    if (dehum > 0) {
      $("dehum").value = "0";
    }
  } catch (err) {
    $("cleanInfo").textContent = "ノイズ除去に失敗しました";
    toast(err.message);
  } finally {
    $("cleanBtn").disabled = false;
  }
}

function revertInput() {
  if (!state.original) return;
  renderInput(
    state.original.blob,
    state.original.samples,
    state.original.name,
    `${state.original.name} ／ 元の音声に戻しました`,
    state.original.uploadName,
  );
  state.cleaned = false;
  $("revertBtn").classList.add("hidden");
  $("cleanInfo").textContent = "元に戻しました（もう一度かけることもできます）";
}

// ------------------------------------------------------------------ 録音
async function startRecording() {
  if (document.body.dataset.streaming === "true") return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    toast("このブラウザはマイク録音に対応していません");
    return;
  }
  if (!window.MediaRecorder) {
    toast("このブラウザは MediaRecorder に対応していません");
    return;
  }

  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
  } catch (err) {
    toast(`マイクを使えません: ${err.message}`);
    return;
  }

  const chunks = [];
  const recorder = new MediaRecorder(state.stream);
  state.recorder = recorder;
  recorder.ondataavailable = (event) => { if (event.data.size) chunks.push(event.data); };
  recorder.onstop = async () => {
    state.stream.getTracks().forEach((track) => track.stop());
    clearInterval(state.recTimer);
    $("recordBtn").classList.remove("on");
    $("recordBtn").textContent = "● マイクで録音";
    $("recordTime").classList.add("hidden");

    const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
    if (!blob.size) {
      toast("録音が空でした");
      return;
    }
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const decoded = await ctx.decodeAudioData(await blob.arrayBuffer());
      await ctx.close();
      await setInputFromBuffer(decoded, `録音 ${clock(state.recSeconds)}`);
    } catch (err) {
      toast(`録音を読めませんでした: ${err.message}`);
    }
  };

  recorder.start();
  state.recSeconds = 0;
  $("recordBtn").classList.add("on");
  $("recordBtn").textContent = "■ 録音を停止";
  $("recordTime").classList.remove("hidden");
  $("recordTime").textContent = "00:00";
  state.recTimer = setInterval(() => {
    state.recSeconds += 1;
    $("recordTime").textContent = clock(state.recSeconds);
    if (state.recSeconds >= 120) recorder.stop();
  }, 1000);
}

// ------------------------------------------------------------------ 変換
function collectForm() {
  const form = new FormData();
  form.append("audio", state.wavBlob, state.uploadName);
  form.append("model", $("modelSelect").value);
  form.append("device", $("deviceSelect").value);
  form.append("ort_cuda", String($("ortDevice").value === "cuda"));
  form.append("pitch", $("pitch").value);
  form.append("speaker", $("speaker").value || "0");
  form.append("rms_mix", $("rmsMix").value);
  form.append("denoise", $("denoise").value);
  form.append("dehum", $("dehum").value);
  form.append("seed", $("seed").value || "1234");
  form.append("output_sr", $("outputSr").value);
  form.append("mode", document.querySelector('input[name="mode"]:checked').value);
  form.append("hop", $("hop").value);
  form.append("context", $("context").value);
  form.append("crossfade", $("crossfade").value);
  form.append("metrics", "true");
  return form;
}

async function startConvert() {
  if (document.body.dataset.streaming === "true") return;
  if (!state.wavBlob) {
    toast("先に音声を選ぶか録音してください");
    return;
  }
  $("convertBtn").disabled = true;
  document.body.dataset.converting = "true";
  $("cancelBtn").classList.remove("hidden");
  $("idlePanel").classList.add("hidden");
  $("resultPanel").classList.add("hidden");
  $("errorBox").classList.add("hidden");
  $("progressPanel").classList.remove("hidden");
  setProgress("送信中", 0);
  state.startedAt = performance.now();

  try {
    const { id } = await api("/api/jobs", { method: "POST", body: collectForm() });
    state.jobId = id;
    pollJob(id);
  } catch (err) {
    failJob(err.message);
  }
}

function setProgress(stage, progress) {
  $("stageText").textContent = stage;
  const pct = Math.max(0, Math.min(1, progress || 0));
  $("progressBar").style.width = `${(pct * 100).toFixed(1)}%`;
  $("progressPct").textContent = `${Math.round(pct * 100)} %`;
  $("elapsedText").textContent = `経過 ${((performance.now() - state.startedAt) / 1000).toFixed(1)} 秒`;
}

function pollJob(jobId) {
  clearInterval(state.poll);
  state.poll = setInterval(async () => {
    try {
      const job = await api(`/api/jobs/${jobId}`);
      setProgress(job.stage, job.progress);
      if (job.status === "done") {
        clearInterval(state.poll);
        finishJob(job);
      } else if (job.status === "error") {
        clearInterval(state.poll);
        failJob(job.error || "変換に失敗しました");
      } else if (job.status === "canceled") {
        clearInterval(state.poll);
        failJob("中断しました");
      }
    } catch (err) {
      clearInterval(state.poll);
      failJob(err.message);
    }
  }, 500);
}

function failJob(message) {
  document.body.dataset.converting = "false";
  $("progressPanel").classList.add("hidden");
  $("errorBox").textContent = message;
  $("errorBox").classList.remove("hidden");
  $("convertBtn").disabled = false;
  $("cancelBtn").classList.add("hidden");
  toast(message);
  refreshHistory();
}

async function finishJob(job) {
  document.body.dataset.converting = "false";
  $("convertBtn").disabled = false;
  $("cancelBtn").classList.add("hidden");
  $("progressPanel").classList.add("hidden");
  $("resultPanel").classList.remove("hidden");
  $("errorBox").classList.add("hidden");

  const url = `${job.audio}?t=${Date.now()}`;
  $("outputAudio").src = url;
  $("downloadLink").href = url;
  $("downloadLink").download = `converted_${job.request.model.replace(/[\\/]/g, "_")}.wav`;

  try {
    const res = await fetch(url);
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const decoded = await ctx.decodeAudioData(await res.arrayBuffer());
    await ctx.close();
    drawWave($("outputWave"), decoded.getChannelData(0));
  } catch (_) {
    drawWave($("outputWave"), null);
  }

  renderMetrics(job);
  await refreshHistory();
}

function renderMetrics(job) {
  const m = job.metrics || {};
  const rows = [];
  const add = (label, value, cls = "") => rows.push(`<tr><th>${label}</th><td class="${cls}">${value}</td></tr>`);

  add("処理時間", `${fmt(job.timings.convert ?? job.timings.process, 2)} 秒（音声 ${fmt(job.input_seconds, 2)} 秒）`);
  if (job.rtf !== null && job.rtf !== undefined) {
    add("実時間比（RTF）", fmt(job.rtf, 2), job.rtf < 1 ? "good" : "bad");
  }
  if (job.request.mode === "chunk") {
    add("区間あたり遅延", `平均 ${fmt(job.latency_mean_ms, 0)} ms / 最大 ${fmt(job.latency_max_ms, 0)} ms`);
    add("ライブ成立（最大遅延 < hop）", job.realtime_ok ? "可" : "不可", job.realtime_ok ? "good" : "bad");
    add("区間数", `${job.chunks}（hop ${job.request.hop} 秒）`);
    if (m.boundary_jump_mean !== undefined) {
      add("継ぎ目の跳び", `平均 ${fmt(m.boundary_jump_mean, 4)} / 最大 ${fmt(m.boundary_jump_max, 4)}`);
    }
  }
  if (m.f0_median_in !== undefined) {
    add("F0 中央値", `${fmt(m.f0_median_in, 1)} Hz → ${fmt(m.f0_median_out, 1)} Hz`);
    add("F0 追従（log F0 の相関）", fmt(m.f0_correlation, 3), m.f0_correlation >= 0.9 ? "good" : "bad");
    add("エネルギー包絡の相関", fmt(m.envelope_correlation, 3), m.envelope_correlation >= 0.9 ? "good" : "bad");
    add("調波性", `${fmt(m.harmonicity_in, 3)} → ${fmt(m.harmonicity_out, 3)}`);
    add("ピーク / クリップ", `${fmt(m.peak_out, 3)} / ${m.clipped ? "あり" : "なし"}`, m.clipped ? "bad" : "good");
  }
  if (m.noise_floor_in_db !== undefined && m.noise_floor_out_db !== undefined) {
    const delta = m.noise_floor_out_db - m.noise_floor_in_db;
    add("ノイズフロア（入力 → 出力）", `${fmt(m.noise_floor_in_db, 1)} → ${fmt(m.noise_floor_out_db, 1)} dBFS`, delta <= -3 ? "good" : "");
  }
  if (job.request.denoise > 0 || job.request.dehum > 0) {
    const hum = job.request.dehum > 0 ? ` / ハム ${job.request.dehum} Hz` : "";
    add("ノイズ除去", `強さ ${fmt(job.request.denoise, 2)}${hum}`);
  }
  add("デバイス", job.request.device.toUpperCase());
  add("特徴抽出", job.request.ort_cuda ? "CUDA 優先" : "CPU");

  $("metricsBody").innerHTML = rows.join("");
}

async function refreshHistory() {
  try {
    const { jobs } = await api("/api/jobs");
    const list = $("historyList");
    if (!jobs.length) {
      list.innerHTML = '<li class="hint">まだありません</li>';
      return;
    }
    list.innerHTML = jobs.map((job) => {
      const time = new Date((job.created_at || 0) * 1000).toLocaleTimeString();
      const mark = { done: "✓", error: "✕", canceled: "—", running: "…", queued: "…" }[job.status] || "?";
      const duration = job.input_seconds ? `${fmt(job.input_seconds, 1)} 秒` : "";
      return `<li data-id="${job.id}">
        <span>${mark} ${job.request.model} <span class="meta">${job.request.mode === "chunk" ? "分割" : "一括"} / ${job.request.device}</span></span>
        <span class="meta">${duration} / ${time}</span>
      </li>`;
    }).join("");
    list.querySelectorAll("li[data-id]").forEach((item) => {
      item.addEventListener("click", () => loadJob(item.dataset.id));
    });
  } catch (_) {
    /* 履歴が取れなくても変換自体には影響しない */
  }
}

async function loadJob(jobId) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    if (job.status === "done") {
      $("idlePanel").classList.add("hidden");
      $("progressPanel").classList.add("hidden");
      await finishJob(job);
    } else if (job.status === "error") {
      failJob(job.error || "変換に失敗しました");
    } else {
      $("progressPanel").classList.remove("hidden");
      setProgress(job.stage, job.progress);
      state.startedAt = performance.now();
      pollJob(job.id);
    }
  } catch (err) {
    toast(err.message);
  }
}

// ------------------------------------------------------------------ 初期化
async function init() {
  try {
    const health = await api("/api/health");
    const badge = $("healthBadge");
    if (health.cuda) {
      badge.textContent = `CUDA: ${health.device}`;
      badge.classList.add("ok");
    } else {
      badge.textContent = "CPU のみ（GPU なし）";
      badge.classList.add("ng");
    }
    $("deviceSelect").value = "auto";
    $("deviceSelect").title = `torch ${health.torch}`;
    $("ortHint").textContent = health.ort_cuda_available
      ? "ONNX CUDA 版を検出しました。開始時に両特徴抽出器の CUDA 利用可否を確認します。"
      : "特徴抽出の GPU 化には、対応する onnxruntime-gpu と CUDA / cuDNN が必要です。";
  } catch (_) {
    $("healthBadge").textContent = "サーバーに接続できません";
    return;
  }

  try {
    const { models } = await api("/api/models");
    state.models = models;
    const select = $("modelSelect");
    select.innerHTML = models.map((m) => `<option value="${m.id}">${m.name}（${m.size_mb} MB）</option>`).join("");
    if (!models.length) {
      select.innerHTML = '<option value="">models/ に .pth がありません</option>';
      $("convertHint").textContent = "models/ に RVC v2 の .pth を置いてください";
    }
    select.addEventListener("change", async () => {
      if (!select.value) return;
      try {
        const info = await api(`/api/models/${select.value}`);
        $("modelBadge").textContent = `${info.sample_rate} Hz / 話者 ${info.speakers}`;
        $("modelBadge").classList.remove("hidden");
      } catch (_) { /* 詳細が取れなくても変換はできる */ }
    });
    if (models.length) select.dispatchEvent(new Event("change"));
    window.dispatchEvent(new Event("vlvc-models-ready"));
  } catch (err) {
    toast(`モデル一覧を取得できません: ${err.message}`);
  }

  // 入力
  const dropzone = $("dropzone");
  dropzone.addEventListener("click", () => $("fileInput").click());
  $("fileInput").addEventListener("change", (event) => setInputFromFile(event.target.files[0]));
  ["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.add("drag");
  }));
  ["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.remove("drag");
  }));
  dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) setInputFromFile(file);
  });

  $("recordBtn").addEventListener("click", () => {
    if (state.recorder && state.recorder.state === "recording") {
      state.recorder.stop();
    } else {
      startRecording();
    }
  });

  $("cleanBtn").addEventListener("click", applyDenoiseToInput);
  $("revertBtn").addEventListener("click", revertInput);

  // 設定
  $("pitch").addEventListener("input", () => { $("pitchValue").textContent = $("pitch").value; });
  $("rmsMix").addEventListener("input", () => { $("rmsValue").textContent = Number($("rmsMix").value).toFixed(2); });
  $("denoise").addEventListener("input", () => { $("denoiseValue").textContent = Number($("denoise").value).toFixed(2); });
  document.querySelectorAll('input[name="mode"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      const chunk = document.querySelector('input[name="mode"]:checked').value === "chunk";
      $("chunkOptions").classList.toggle("hidden", !chunk);
    });
  });

  $("convertBtn").addEventListener("click", startConvert);
  $("cancelBtn").addEventListener("click", async () => {
    if (!state.jobId) return;
    try {
      await api(`/api/jobs/${state.jobId}/cancel`, { method: "POST" });
      $("stageText").textContent = "中断しています…";
    } catch (err) {
      toast(err.message);
    }
  });

  window.addEventListener("resize", () => {
    if (state.buffer) drawWave($("inputWave"), state.buffer);
  });

  await refreshHistory();
}

init();
