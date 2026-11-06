"""Fetch a historical checkpoint and verify its release digest before installation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download(asset: dict, destination: Path) -> Path:
    name = asset['name']
    if Path(name).name != name or name in {'.', '..'}:
        raise ValueError('Asset name must be a basename')
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / name
    if target.exists():
        if target.stat().st_size == asset['bytes'] and sha256(target) == asset['sha256']:
            print(f'Already verified: {name}')
            return target
        raise ValueError(f'{target} already exists but does not match the manifest; move it before retrying')
    fd, temporary = tempfile.mkstemp(prefix=name + '.', suffix='.part', dir=destination)
    part = Path(temporary)
    try:
        request = urllib.request.Request(asset.get('mirror_url', asset['url']), headers={'User-Agent': 'rsna-knee-mri-artifact-downloader/1.0'})
        digest = hashlib.sha256()
        total = 0
        with os.fdopen(fd, 'wb') as output, urllib.request.urlopen(request, timeout=90) as response:
            for block in iter(lambda: response.read(1024 * 1024), b''):
                output.write(block)
                digest.update(block)
                total += len(block)
        if total != asset['bytes'] or digest.hexdigest() != asset['sha256']:
            raise ValueError(f'Size or SHA-256 verification failed for {name}')
        # Do not replace a file another process installed while this download ran.
        os.link(part, target)
        print(f'Verified: {name} ({total:,} bytes)')
        return target
    finally:
        part.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--list', action='store_true', help='List assets; no downloads')
    group.add_argument('--name', action='append', help='Exact asset name; repeat for multiple files')
    group.add_argument('--all', action='store_true', help='Download the entire historical release, including failed SSL controls')
    parser.add_argument('--directory', type=Path, default=ROOT / 'models')
    args = parser.parse_args()
    assets = json.loads((ROOT / 'models/manifest.json').read_text())['assets']
    by_name = {a['name']: a for a in assets}
    if args.list:
        for a in assets:
            print(f"{a['name']:52} {a['bytes'] / 1e6:7.1f} MB  {a['status']}")
        print(f"Total: {sum(a['bytes'] for a in assets) / 1e9:.2f} GB; source: models/manifest.json")
        return
    names = list(by_name) if args.all else list(dict.fromkeys(args.name))
    unknown = set(names) - by_name.keys()
    if unknown:
        parser.error('Unknown asset(s): ' + ', '.join(sorted(unknown)))
    for name in names:
        download(by_name[name], args.directory)


if __name__ == '__main__':
    main()
