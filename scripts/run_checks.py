"""Run publication checks without competition data or pretrained downloads."""
from pathlib import Path
import os
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = [
    ['scripts/check_repository.py'],
    ['-m', 'unittest', 'discover', '-v', '-s', 'tests'],
    ['-m', 'unittest', 'discover', '-v', '-s', 'kaggle/roi_cache', '-p', 'test_*.py'],
    ['-m', 'unittest', 'discover', '-v', '-s', 'kaggle/armc_roi', '-p', 'test_*.py'],
    ['-m', 'unittest', 'discover', '-v', '-s', 'kaggle/model_research', '-p', 'test_*.py'],
    ['kaggle/portfolio/test_window_headers.py'],
    ['kaggle/portfolio/smoke.py'],
    ['-m', 'torch.distributed.run', '--nnodes=1', '--master-addr=127.0.0.1', '--master-port={port}', '--nproc_per_node=2', 'kaggle/portfolio/ddp_smoke.py'],
    ['run_web_annotator.py', '--help'],
]


def main():
    env = dict(os.environ, MPLBACKEND='Agg', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', WANDB_MODE='disabled')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    for command in COMMANDS:
        command = [arg.replace('{port}', str(port)) for arg in command]
        print('\nRunning:', ' '.join(command), flush=True)
        subprocess.run([sys.executable, *command], cwd=ROOT, env=env, check=True, timeout=300)
    print('\nAll checks passed. Data-dependent skips do not validate real MRI inference.')


if __name__ == '__main__':
    main()
