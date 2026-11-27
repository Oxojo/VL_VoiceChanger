const form = document.querySelector('#probeForm');
const status = document.querySelector('#status');
const run = document.querySelector('#run');
let report = null;

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  run.disabled = true;
  report = null;
  document.querySelector('#results').hidden = true;
  status.textContent = 'モデル準備・ウォームアップ・100フレームの計測中…';
  try {
    const response = await fetch('/api/streamvc/probe', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({device: document.querySelector('#device').value,
        threads: Number(document.querySelector('#threads').value)}),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '計測要求を受け付けられませんでした。');
    report = data;
    const ms = data.neural_ms;
    const metrics = [
      ['実行環境', `${data.device} / ${data.threads} thread / PyTorch ${data.torch}`],
      ['ニューラル計算 平均 / p95', `${ms.mean.toFixed(2)} / ${ms.p95.toFixed(2)} ms`],
      ['20msを超えたフレーム', `${data.frames_over_20ms} / ${data.frames}`],
      ['real-time factor', data.real_time_factor.toFixed(2)],
      ['保持状態', `${(data.state_bytes / 1024).toFixed(1)} KiB / ${data.state_size_constant ? '一定' : '変動あり'}`],
      ['一括処理との最大誤差', data.full_vs_stream_max_abs_error.toExponential(2)],
    ];
    const rows = document.querySelector('#rows');
    rows.replaceChildren();
    for (const [label, value] of metrics) {
      const tr = document.createElement('tr');
      for (const text of [label, value]) {
        const td = document.createElement('td'); td.textContent = text; tr.append(td);
      }
      rows.append(tr);
    }
    document.querySelector('#results').hidden = false;
    status.textContent = data.real_time_factor >= 1
      ? '計測完了。この条件では平均処理時間が20ms以上で、連続入力に追いつきません。'
      : '計測完了。平均処理時間は20ms未満です。p95・超過フレーム数も確認してください。';
  } catch (error) {
    status.textContent = `計測できませんでした: ${error.message}`;
  } finally {
    run.disabled = false;
  }
});
document.querySelector('#download').addEventListener('click', () => {
  if (!report) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], {type: 'application/json'}));
  const link = document.createElement('a');
  link.href = url; link.download = 'vlvc-streamvc-probe.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
