"""Check source syntax, notebook hygiene, local Markdown links and artifact manifests."""
import ast
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = [p for p in ROOT.rglob('*') if p.is_file() and not any(x.startswith('.') or x == '__pycache__' for x in p.relative_to(ROOT).parts)]
    errors = []
    for path in files:
        if path.suffix == '.py':
            try: ast.parse(path.read_text(), filename=str(path.relative_to(ROOT)))
            except SyntaxError as error: errors.append(str(error))
        elif path.suffix == '.ipynb':
            for cell in json.loads(path.read_text())['cells']:
                if cell['cell_type'] == 'code' and (cell.get('outputs') or cell.get('execution_count') is not None):
                    errors.append(f'{path.name}: notebook output/execution state retained')
        elif path.suffix == '.md':
            for link in re.findall(r'\]\(([^\s)]+)(?:\s+"[^"]*")?\)', path.read_text()):
                if '://' in link or link.startswith(('#', 'mailto:')): continue
                target = unquote(link.split('#')[0])
                if target and not (path.parent / target).exists(): errors.append(f'{path.relative_to(ROOT)}: broken link {link}')
    model_manifest = json.loads((ROOT/'models/manifest.json').read_text())
    assets = model_manifest['assets']
    assert len({a['name'] for a in assets}) == len(assets)
    for asset in assets:
        assert re.fullmatch('[0-9a-f]{64}', asset['sha256']) and asset['bytes'] > 0
        assert asset['url'].startswith('https://github.com/XERO47/rsna-knee-mri/releases/download/')
    rows = json.loads((ROOT/'evidence/submissions.json').read_text())
    assert len(rows) == len({r['ref'] for r in rows}) == 30
    assert all(r['status'] == 'COMPLETE' for r in rows)
    assert max(float(r['publicScore']) for r in rows) == .937
    assert max(float(r['privateScore']) for r in rows) == .920
    for item in json.loads((ROOT/'evidence/source_manifest.json').read_text()):
        path = ROOT/item['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            errors.append(f"Source manifest mismatch: {item['path']}")
    if errors: raise SystemExit('\n'.join(errors))
    print(f'Syntax, notebook, Markdown and manifest checks passed ({len(files)} files; {len(assets)} model assets).')


if __name__ == '__main__': main()
