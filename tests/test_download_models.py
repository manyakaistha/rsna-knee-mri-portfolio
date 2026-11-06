import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import io

spec = importlib.util.spec_from_file_location('download_models', Path(__file__).resolve().parents[1] / 'scripts/download_models.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DownloadIntegrityTest(unittest.TestCase):
    def test_verified_download_and_idempotent_reuse(self):
        data = b'synthetic checkpoint fixture'
        asset = dict(name='fixture.pt', bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), url='https://example.invalid/fixture')
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            with patch.object(module.urllib.request, 'urlopen', return_value=io.BytesIO(data)) as request:
                result = module.download(asset, dest)
                self.assertEqual(result.read_bytes(), data)
                self.assertEqual(module.download(asset, dest), result)
                self.assertEqual(request.call_count, 1)
            self.assertEqual(list(dest.glob('*.part')), [])

    def test_corrupt_download_never_installed(self):
        asset = dict(name='fixture.pt', bytes=4, sha256=hashlib.sha256(b'good').hexdigest(), url='https://example.invalid/fixture')
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            with patch.object(module.urllib.request, 'urlopen', return_value=io.BytesIO(b'evil')):
                with self.assertRaises(ValueError): module.download(asset, dest)
            self.assertEqual(list(dest.iterdir()), [])

    def test_existing_mismatch_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory); (dest/'fixture.pt').write_bytes(b'keep')
            with self.assertRaises(ValueError):
                module.download(dict(name='fixture.pt', bytes=0, sha256='wrong', url='unused'), dest)
            self.assertEqual((dest/'fixture.pt').read_bytes(), b'keep')
