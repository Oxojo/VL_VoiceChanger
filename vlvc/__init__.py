"""VL_VoiceChanger のクロスプラットフォーム CLI。

`python -m vlvc <コマンド>` で動く。Windows / macOS / Linux のどれでも、
PowerShell を必要としない。変換そのものは ``engine``（Python + torch/ONNX）が担当し、
このパッケージは環境確認・起動・設定投入・計測・レポート作成を受け持つ。

主なコマンド::

    vlvc setup          仮想環境と依存関係を用意する
    vlvc doctor         機材・実行環境を確認してレポートを出す
    vlvc convert        WAV を変換する
    vlvc serve          ブラウザ UI を起動する
    vlvc quality        変換前後を数値で比較する
    vlvc selftest       変換〜計測をまとめて実行しレポートを出す
    vlvc launch         VC Client を起動して GUI 設定項目を表示する
    vlvc profile        VC Client のプロファイルを適用する
    vlvc register-model VC Client に RVC モデルを登録する
    vlvc stability      連続使用テストを記録する
"""

__version__ = "0.1.0"
