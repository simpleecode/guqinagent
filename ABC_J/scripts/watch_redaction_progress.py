"""Display redaction progress without reading trajectories or making API calls."""
import argparse
import json
from pathlib import Path
import sys
import time

DEFAULT = Path(__file__).resolve().parents[1] / 'agent_training/two_stage_audited/latest_runtime_redaction'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    parser.add_argument('--interval', type=float, default=2)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    try:
        while True:
            values = []
            for path in sorted(args.directory.glob('worker_*/reasoning_redaction_progress.json')):
                try:
                    values.append((path.parent.name, json.loads(path.read_text())))
                except (FileNotFoundError, json.JSONDecodeError):
                    continue
            total = sum(r.get('requested', 0) for _, r in values)
            done = sum(r.get('completed', 0) for _, r in values)
            failed = sum(r.get('failed', 0) for _, r in values)
            ratio = done / total if total else 0
            filled = min(32, int(32 * ratio))
            bar = '█' * filled + '░' * (32 - filled)
            lines = [f'GLM 脱敏 [{bar}] {ratio:6.2%}  {done:,}/{total:,}  失败 {failed}']
            for name, r in values:
                lines.append(f'{name}: {r.get("completed", 0):4}/{r.get("requested", 0)}  失败 {r.get("failed", 0)}  {r.get("status", "未知")}')
            complete = (args.directory / 'completion_report.json').exists()
            if complete:
                lines.append('脱敏、校验、合并及训练格式导出已完成。')
            elif total and done == total and not failed:
                lines.append('分片脱敏完成；等待主流程校验、合并及导出。')
            elif not values:
                lines.append('尚无分片进度文件。')
            if sys.stdout.isatty() and not args.once:
                print('\033[2J\033[H', end='')
            print('\n'.join(lines), flush=True)
            if args.once or complete:
                break
            time.sleep(max(.2, args.interval))
    except KeyboardInterrupt:
        print('\n已退出进度查看；脱敏任务继续运行。')


if __name__ == '__main__':
    main()
