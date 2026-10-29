"""連続使用テスト（旧 ``Test-Stability.ps1``）。

設計書 6 章 Phase 3「安定性検証」に対応する。指定時間、VC Client のプロセス・
GUI ポート・GPU 使用率を一定間隔で記録し、CSV（生ログ）と Markdown（判定用レポート）を出す。

実行中に次のキーを押すと、その時刻の事象を記録できる::

    D … 音切れがあった
    S … 変換が止まった / フリーズした
    N … 声が崩れた
    M … 自由記述のメモ
    Q … テストを途中終了する

「音が明瞭に聞こえるか」は機械では判定できない。評価担当 2 名の聴取結果は
docs/04-verification.md の評価シートに記入する。
"""

from __future__ import annotations

import csv
import datetime as dt
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from . import sysinfo, ui, vcclient
from .config import Settings
from .keys import KeyListener

CSV_COLUMNS = (
    ("index", "Index"),
    ("elapsed", "ElapsedSec"),
    ("clock", "Clock"),
    ("process_ok", "ProcessOk"),
    ("cpu_pct", "CpuPercent"),
    ("memory_mb", "MemoryMB"),
    ("threads", "Threads"),
    ("handles", "Handles"),
    ("gpu_util", "GpuUtilPct"),
    ("gpu_mem_mb", "GpuMemMB"),
    ("gpu_temp", "GpuTempC"),
    ("port_open", "PortOpen"),
    ("http_ok", "HttpOk"),
    ("http_status", "HttpStatus"),
    ("event", "Event"),
)

OPERATOR_KINDS = ("音切れ", "停止", "変換停止", "声の崩れ")


def stats(values: Iterable[float | None]) -> dict[str, float] | None:
    """平均・最小・最大を求める（None は除外）。値が無ければ None。"""
    data = [float(v) for v in values if v is not None]
    if not data:
        return None
    return {
        "avg": round(sum(data) / len(data), 1),
        "min": round(min(data), 1),
        "max": round(max(data), 1),
        "n": len(data),
    }


def _handle_count(proc: Any) -> int | None:
    for name in ("num_handles", "num_fds"):
        method = getattr(proc, name, None)
        if method is None:
            continue
        try:
            return int(method())
        except Exception:  # noqa: BLE001 - 権限が無い環境では取得しない
            continue
    return None


def _http_probe(url: str, timeout: float) -> tuple[bool, str]:
    request = urllib.request.Request(url, headers={"Accept": "*/*"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return True, str(response.status)
    except urllib.error.HTTPError as exc:
        return False, f"error: HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001 - 接続不可も記録対象
        return False, f"error: {exc}"


def build_markdown(
    *,
    generated_at: str,
    profile: str,
    minutes: int,
    actual_seconds: float,
    sample_seconds: int,
    sample_count: int,
    process_pattern: str,
    url: str,
    csv_path: Path,
    proc_uptime: float,
    http_uptime: float,
    proc_down: int,
    port_down: int,
    http_fail: int,
    proc_stats: dict[str, float] | None,
    gpu_stats: dict[str, float] | None,
    temp_stats: dict[str, float] | None,
    mem_stats: dict[str, float] | None,
    incidents: list[dict[str, Any]],
    operator_events: int,
) -> str:
    lines: list[str] = []
    add = lines.append

    add("# 安定性検証レポート")
    add("")
    add(f"- 実施日時: {generated_at}")
    add(f"- プロファイル: {profile or '（未指定）'}")
    add(f"- 目標時間: {minutes} 分 / 実測: {actual_seconds} 秒")
    add(f"- サンプリング間隔: {sample_seconds} 秒（{sample_count} サンプル）")
    add(f"- 監視対象: VC Client プロセス（自動判定: `{process_pattern}`）/ `{url}`")
    add(f"- 生ログ: `{csv_path}`")
    add("")
    add("## 自動計測の結果")
    add("")
    add("| 指標 | 値 |")
    add("| :--- | :--- |")
    add(f"| プロセス稼働率 | {proc_uptime} % |")
    add(f"| GUI 応答率 | {http_uptime} % |")
    add(f"| プロセス消失サンプル数 | {proc_down} |")
    add(f"| ポート無応答サンプル数 | {port_down} |")
    add(f"| HTTP 失敗サンプル数 | {http_fail} |")
    if proc_stats:
        add(f"| CPU 使用率（平均 / 最大） | {proc_stats['avg']} % / {proc_stats['max']} % |")
    else:
        add("| CPU 使用率 | 取得できず |")
    if gpu_stats:
        add(f"| GPU 使用率（平均 / 最大） | {gpu_stats['avg']} % / {gpu_stats['max']} % |")
    else:
        add("| GPU 使用率 | 取得できず（nvidia-smi なし） |")
    if temp_stats:
        add(f"| GPU 温度（最大） | {temp_stats['max']} ℃ |")
    if mem_stats:
        add(f"| メモリ（平均 / 最大） | {mem_stats['avg']} MB / {mem_stats['max']} MB |")
    add(f"| 操作者が記録した事象 | {operator_events} 件 |")
    add("")
    add("## 記録された事象")
    add("")
    if not incidents:
        add("自動検出・操作者記録ともに事象なし。")
    else:
        add("| 経過 | 時刻 | 種別 | 内容 |")
        add("| :--- | :--- | :--- | :--- |")
        for entry in incidents:
            add(
                f"| {entry['elapsed']} 秒 | {entry['clock']} | {entry['kind']} | {entry['detail']} |"
            )
    add("")
    add("## 聴取評価（評価担当が記入）")
    add("")
    add("機械では判定できないため、評価担当 2 名程度が次を記入する。詳細は docs/04-verification.md の評価シートを使う。")
    add("")
    add("| 発話パターン | 評価担当1 | 評価担当2 | コメント |")
    add("| :--- | :--- | :--- | :--- |")
    add("| 通常発話 |  |  |  |")
    add("| 早口 |  |  |  |")
    add("| 感情を込めた発話 |  |  |  |")
    add("")
    add("## MVP 判定（設計書 2.3）")
    add("")
    add("- [ ] 10 分間の連続発話中に明らかな停止がなかった")
    add("- [ ] 頻繁な音切れがなかった")
    add("- [ ] 著しい声の崩れがなかった")
    add("- [ ] 評価担当 2 名が通常発話・早口・感情発話を聞き取れた")
    add("- [ ] 会話のやり取りが可能な遅延だった（体感評価）")
    add("")
    add("判定: **未判定**（上記を記入して確定させる）")
    add("")
    add("## 所見・次アクション")
    add("")
    add("- ")
    add("")
    add("> 自動計測は「ソフトが生きているか」しか見ていない。音切れ・声質は必ず人の耳で判定すること。")
    add("")
    return "\n".join(lines)


def run(
    settings: Settings,
    minutes: int = 0,
    sample_seconds: int = 0,
    profile: str = "",
    process_pattern: str = "",
    url: str = "",
    timeout: int = 5,
    allow_not_running: bool = False,
) -> int:
    if minutes <= 0:
        minutes = settings.int("stabilityTest.durationMinutes", 10)
    if sample_seconds <= 0:
        sample_seconds = settings.int("stabilityTest.sampleSeconds", 5)
    if not process_pattern:
        process_pattern = str(
            settings.get("stabilityTest.processName", "main,MMVC_Client*,MMVCServerSIO*,vcclient*")
        )
    port = settings.port
    if not url:
        url = f"http://localhost:{port}{settings.health_path}"

    report_dir = settings.reports_dir
    log_dir = settings.logs_dir
    for directory in (report_dir, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    ui.header("VL_VoiceChanger / 安定性検証")
    print(f" 予定時間: {minutes} 分 / サンプリング: {sample_seconds} 秒")
    print(f" 監視対象: VC Client プロセス（自動判定） / {url}")

    try:
        psutil = __import__("psutil")
    except ImportError:
        ui.err("psutil が必要です。`vlvc setup` か `pip install psutil` を実行してください。")
        return 1

    # ---------------------------------------------------------------- 事前確認
    initial = vcclient.find_processes(settings.vc_root, process_pattern)
    server_pid = vcclient.listener_pid(port)
    if server_pid is not None:
        ui.info(f"サーバープロセス: PID {server_pid}（ポート {port} の待受）")
    servers = [p for p in initial if "voice-changer-native-client" not in (p.name() or "").lower()]
    if len(servers) > 1:
        ui.warn(f"VC Client のサーバープロセスが {len(servers)} 個あります。多重起動の可能性があります。")

    port_open = vcclient.port_open(port, settings.host, timeout=1.0)
    if not initial and not port_open:
        ui.err(f"VC Client が起動していません（プロセスもポート {port} も応答なし）。")
        ui.info("先に `vlvc launch` を実行し、[start] を押して変換を開始してください。")
        if not allow_not_running:
            ui.info("監視だけ行いたい場合は --allow-not-running を付けて実行してください。")
            return 1
        ui.warn("--allow-not-running のため続行します。全サンプルが異常記録になります。")

    if port_open:
        valid, total = vcclient.module_status(settings)
        if total and valid < total:
            ui.warn(f"VC Client の必須モジュールが {total - valid} 件そろっていません（この状態では RVC は動きません）。")
            ui.info("VC Client を再起動してダウンロードを完了させてください（docs/03-troubleshooting.md）。")
        elif total:
            ui.ok(f"必須モジュール: {total} 件すべて正常")

    # ---------------------------------------------------------------- ループ
    ui.step(f"計測開始（{dt.datetime.now().strftime('%H:%M:%S')}）")
    ui.note(" 操作: [D] 音切れ  [S] 停止  [N] 声の崩れ  [M] メモ  [Q] 終了", "cyan")

    samples: list[dict[str, Any]] = []
    incidents: list[dict[str, Any]] = []
    started = time.monotonic()
    total_seconds = minutes * 60
    last_proc_ok: bool | None = None
    last_port_ok: bool | None = None
    last_http_ok: bool | None = None
    previous_cpu: float | None = None
    previous_at: float | None = None
    cpu_count = psutil.cpu_count(logical=True) or 1
    stop = False

    def add_incident(kind: str, detail: str, elapsed: float) -> None:
        entry = {
            "elapsed": round(elapsed, 1),
            "clock": dt.datetime.now().strftime("%H:%M:%S"),
            "kind": kind,
            "detail": detail,
        }
        incidents.append(entry)
        print()
        ui.err(f"!! [{entry['clock']}] {kind} — {detail}")

    listener = KeyListener()
    if not listener.enabled:
        ui.info("対話コンソールが無いためキー入力の記録は無効です。")

    index = 0
    try:
        while not stop and time.monotonic() - started < total_seconds:
            index += 1
            now_wall = dt.datetime.now()
            elapsed = time.monotonic() - started

            # --- プロセス
            procs = vcclient.find_processes(settings.vc_root, process_pattern)
            proc = None
            if server_pid is not None:
                proc = next((p for p in procs if p.pid == server_pid), None)
            if proc is None and procs:
                proc = max(procs, key=vcclient._safe_memory)
            proc_ok = proc is not None
            cpu_pct = mem_mb = threads = handles = None
            if proc is not None:
                try:
                    mem_mb = round(proc.memory_info().rss / 1024**2, 1)
                    threads = proc.num_threads()
                    handles = _handle_count(proc)
                    total_time = sum(proc.cpu_times()[:2])
                    if previous_cpu is not None and previous_at is not None:
                        delta = time.monotonic() - previous_at
                        if delta > 0:
                            cpu_pct = round((total_time - previous_cpu) / delta / cpu_count * 100, 1)
                    previous_cpu = total_time
                    previous_at = time.monotonic()
                except Exception:  # noqa: BLE001 - プロセスが消えた直後など
                    proc_ok = False
                    previous_cpu = previous_at = None
            else:
                previous_cpu = previous_at = None

            # --- ポート / HTTP
            port_ok = vcclient.port_open(port, settings.host, timeout=0.8)
            http_ok = False
            http_status = ""
            if port_ok:
                http_ok, http_status = _http_probe(url, timeout)

            # --- GPU
            gpu = sysinfo.gpu_sample()

            # --- 状態遷移
            if last_proc_ok is not None and last_proc_ok != proc_ok:
                if proc_ok:
                    add_incident("プロセス復帰", "VC Client のプロセスを再検出", elapsed)
                else:
                    add_incident("プロセス消失", "VC Client のプロセスがすべて終了した", elapsed)
            if last_port_ok is not None and last_port_ok != port_ok:
                if port_ok:
                    add_incident("GUI 復帰", f"ポート {port} が応答を再開", elapsed)
                else:
                    add_incident("GUI 停止", f"ポート {port} が応答しなくなった", elapsed)
            if last_http_ok is not None and last_http_ok != http_ok:
                if http_ok:
                    add_incident("HTTP 復帰", f"{url} が応答を再開", elapsed)
                else:
                    add_incident("HTTP 失敗", f"{url} の取得に失敗", elapsed)
            last_proc_ok, last_port_ok, last_http_ok = proc_ok, port_ok, http_ok

            row: dict[str, Any] = {
                "index": index,
                "elapsed": round(elapsed, 1),
                "clock": now_wall.strftime("%H:%M:%S"),
                "process_ok": proc_ok,
                "cpu_pct": cpu_pct,
                "memory_mb": mem_mb,
                "threads": threads,
                "handles": handles,
                "gpu_util": gpu["util_pct"] if gpu else None,
                "gpu_mem_mb": gpu["mem_mb"] if gpu else None,
                "gpu_temp": gpu["temp_c"] if gpu else None,
                "port_open": port_ok,
                "http_ok": http_ok,
                "http_status": http_status,
                "event": "",
            }
            samples.append(row)

            cpu_text = f"{cpu_pct:5.1f}%" if cpu_pct is not None else "    - "
            gpu_text = f"{gpu['util_pct']:3.0f}% {gpu['mem_mb']:5.0f}MB" if gpu else "    -        "
            state = "OK  " if (proc_ok and http_ok) else ("WARN" if port_ok else "NG  ")
            remain = max(0, int(total_seconds - (time.monotonic() - started)))
            print(
                f"  [{index:3d}] {now_wall.strftime('%H:%M:%S')} {state}  "
                f"CPU {cpu_text}  GPU {gpu_text}  MEM {(mem_mb or 0):7.1f} MB  残 {remain:3d}s"
            )

            # --- 次のサンプルまでキー入力を受け付ける
            next_at = time.monotonic() + sample_seconds
            while time.monotonic() < next_at and not stop:
                key = listener.poll()
                if key is None:
                    time.sleep(0.15)
                    continue
                now_elapsed = time.monotonic() - started
                if key == "D":
                    add_incident("音切れ", "操作者が音切れを検知", now_elapsed)
                    row["event"] = "音切れ"
                elif key == "S":
                    add_incident("変換停止", "操作者が停止を検知", now_elapsed)
                    row["event"] = "停止"
                elif key == "N":
                    add_incident("声の崩れ", "操作者が声質の崩れを検知", now_elapsed)
                    row["event"] = "声の崩れ"
                elif key == "M":
                    memo = listener.read_line("  メモ: ")
                    add_incident("メモ", memo, now_elapsed)
                    row["event"] = f"メモ: {memo}"
                elif key == "Q":
                    add_incident("手動終了", "操作者がテストを終了", now_elapsed)
                    row["event"] = "終了"
                    stop = True
                else:
                    time.sleep(0.15)
    except KeyboardInterrupt:
        ui.warn("Ctrl+C で計測を終了しました。")
    finally:
        listener.close()
    actual_seconds = round(time.monotonic() - started, 1)

    # ---------------------------------------------------------------- 集計
    ui.step("集計")
    proc_stats = stats(row["cpu_pct"] for row in samples)
    gpu_stats = stats(row["gpu_util"] for row in samples)
    mem_stats = stats(row["memory_mb"] for row in samples)
    temp_stats = stats(row["gpu_temp"] for row in samples)

    sample_count = len(samples)
    proc_down = sum(1 for row in samples if not row["process_ok"])
    port_down = sum(1 for row in samples if not row["port_open"])
    http_fail = sum(1 for row in samples if row["port_open"] and not row["http_ok"])
    operator_events = sum(1 for entry in incidents if entry["kind"] in OPERATOR_KINDS)
    proc_uptime = round((sample_count - proc_down) / sample_count * 100, 1) if sample_count else 0.0
    http_uptime = (
        round((sample_count - port_down - http_fail) / sample_count * 100, 1) if sample_count else 0.0
    )

    ui.info(f"サンプル数: {sample_count} / 実測時間: {actual_seconds} 秒")
    ui.info(f"プロセス稼働率: {proc_uptime} %")
    ui.info(f"GUI 応答率: {http_uptime} %")
    if proc_stats:
        ui.info(f"CPU 平均/最大: {proc_stats['avg']} % / {proc_stats['max']} %")
    if gpu_stats:
        ui.info(f"GPU 平均/最大: {gpu_stats['avg']} % / {gpu_stats['max']} %")
    ui.info(f"操作者が記録した事象: {operator_events} 件")

    # ---------------------------------------------------------------- 出力
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = log_dir / f"stability-{stamp}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([label for _, label in CSV_COLUMNS])
        for row in samples:
            writer.writerow([row[key] if row[key] is not None else "" for key, _ in CSV_COLUMNS])

    markdown = build_markdown(
        generated_at=dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        profile=profile,
        minutes=minutes,
        actual_seconds=actual_seconds,
        sample_seconds=sample_seconds,
        sample_count=sample_count,
        process_pattern=process_pattern,
        url=url,
        csv_path=csv_path,
        proc_uptime=proc_uptime,
        http_uptime=http_uptime,
        proc_down=proc_down,
        port_down=port_down,
        http_fail=http_fail,
        proc_stats=proc_stats,
        gpu_stats=gpu_stats,
        temp_stats=temp_stats,
        mem_stats=mem_stats,
        incidents=incidents,
        operator_events=operator_events,
    )
    md_path = report_dir / f"stability-{stamp}.md"
    md_path.write_text(markdown, encoding="utf-8")

    ui.step("出力")
    ui.ok(f"レポート: {md_path}")
    ui.ok(f"生ログ  : {csv_path}")
    ui.note("次: レポートの「聴取評価」「MVP 判定」を埋め、docs/04-verification.md に転記してください。")
    return 0
