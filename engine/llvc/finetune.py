"""RVC teacher pairs -> LLVC fine-tuning -> selectable voice. Run --help."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='render paired teacher audio from an RVC model')
    prepare.add_argument('--teacher', type=Path, required=True)
    prepare.add_argument('--audio-dir', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    prepare.add_argument('--pitch', type=int, default=0)
    prepare.add_argument('--speaker', type=int, default=0)
    prepare.add_argument('--rms-mix', type=float, default=.25)
    prepare.add_argument('--clip-seconds', type=float, default=4)
    prepare.add_argument('--val-fraction', type=float, default=.1)
    prepare.add_argument('--seed', type=int, default=1234)
    prepare.add_argument('--resume', action='store_true')
    train = commands.add_parser('train', help='fine-tune pretrained LLVC on the paired dataset')
    train.add_argument('--dataset', type=Path, required=True)
    train.add_argument('--run-dir', type=Path, required=True)
    train.add_argument('--base', type=Path)
    train.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    train.add_argument('--steps', type=int, default=10000)
    train.add_argument('--batch-size', type=int, default=1)
    train.add_argument('--segment', type=int, default=8192)
    train.add_argument('--context', type=int, default=8192)
    train.add_argument('--lr', type=float, default=1e-4)
    train.add_argument('--save-every', type=int, default=100)
    train.add_argument('--eval-every', type=int, default=250)
    train.add_argument('--unfreeze-encoder', action='store_true')
    train.add_argument('--seed', type=int, default=1234)
    train.add_argument('--amp', action='store_true')
    train.add_argument('--resume', action='store_true')
    export = commands.add_parser('export', help='install an auditioned checkpoint as a selectable voice')
    export.add_argument('--checkpoint', type=Path, required=True)
    export.add_argument('--voice-id', required=True)
    export.add_argument('--name', required=True)
    args = vars(parser.parse_args())
    command = args.pop('command')
    try:
        if command == 'prepare':
            from .distill import prepare_pairs
            result = prepare_pairs(**args)
        elif command == 'train':
            from .training import fine_tune
            args['freeze_encoder'] = not args.pop('unfreeze_encoder')
            result = fine_tune(**args)
        else:
            from .training import export_voice
            result = export_voice(**args)
        print(f'保存先: {result}')
    except KeyboardInterrupt:
        print('中断しました。trainは最後に保存されたlatest.pthから--resumeで再開できます。', file=sys.stderr)
        raise SystemExit(130)
    except (ValueError,OSError,RuntimeError) as exc:
        parser.exit(1, f'エラー: {exc}\n')


if __name__ == '__main__':
    main()
