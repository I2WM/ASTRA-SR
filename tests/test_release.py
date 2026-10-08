"""Small regression tests for public download/index plumbing; no GPU or network."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_utils as release
from scripts import prepare_dataset as download


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'archives').mkdir()
        protocol = self.root / 'x2_dataset_protocol_lrdegrade_v2.json'
        protocol.write_text('{}')
        self.protocol_hash = release.sha256(protocol)
        self.rows, self.sums = [], {}
        for split in ('val', 'train'):
            paths = {role: f'{split}/real/{role}/{split}.fits' for role in release.ROLES}
            self.rows.append({'source_id': split, 'split': split, 'kind': 'real',
                              'protocol_sha256': self.protocol_hash, 'paths': paths})
            archive = self.root / 'archives' / f'{split}-real-00000-of-00001.tar.gz'
            with tarfile.open(archive, 'w:gz') as stream:
                for relative in paths.values():
                    payload = b'path-plumbing fixture (not scientific FITS data)'
                    member = tarfile.TarInfo(relative)
                    member.size = len(payload)
                    stream.addfile(member, io.BytesIO(payload))
            self.sums[archive.name] = release.sha256(archive)
        index = self.root / 'dataset_index.jsonl'
        index.write_text(''.join(json.dumps(r) + '\n' for r in self.rows), encoding='utf-8')
        sums = self.root / 'SHA256SUMS.txt'
        sums.write_text(''.join(f'{value}  {name}\n' for name,value in self.sums.items()))
        notices = {}
        for name in ('LICENSE', 'LICENSE_SCOPE.md', 'DATA_SOURCES.md', 'THIRD_PARTY_NOTICES.md'):
            path = self.root / name
            path.write_text('Test-only license/source fixture: ' + name, encoding='utf-8')
            notices[name] = release.sha256(path)
        for module in (release, download):
            for key, value in [('COUNTS', {'train': 1, 'val': 1}), ('INDEX_SHA', release.sha256(index)),
                               ('PROTOCOL_SHA', self.protocol_hash), ('SUMS_SHA', release.sha256(sums)),
                               ('LICENSE_FILES', notices)]:
                if hasattr(module, key):
                    self.enterContext(patch.object(module, key, value))

    def run_download(self, *extra):
        with patch.object(sys, 'argv', ['prepare_dataset.py', '--local-only', '--local-dir',
                                       str(self.root), '--splits', 'val', *extra]), contextlib.redirect_stdout(io.StringIO()):
            download.main()

    def test_only_requested_split_and_rebased_index(self):
        self.run_download()
        self.assertFalse((self.root / 'extracted/train').exists())
        index = self.root / 'records_val.jsonl'
        rows = [json.loads(line) for line in index.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['split'], 'val')
        self.assertTrue(all(Path(path).is_file() for path in rows[0]['paths'].values()))
        self.assertEqual(len(list((self.root / 'archives').glob('*.tar.gz'))), 2)

    def test_missing_requested_shard_is_an_error(self):
        (self.root / 'archives/val-real-00000-of-00001.tar.gz').unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_download()
        self.assertFalse((self.root / 'records_val.jsonl').exists())

    def test_corrupt_archive_is_not_extracted(self):
        with (self.root / 'archives/val-real-00000-of-00001.tar.gz').open('ab') as f:
            f.write(b'corruption')
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            self.run_download()
        self.assertFalse((self.root / 'extracted').exists())

    def test_requested_only_archive_removal(self):
        self.run_download('--remove-archives')
        self.assertFalse((self.root / 'archives/val-real-00000-of-00001.tar.gz').exists())
        self.assertTrue((self.root / 'archives/train-real-00000-of-00001.tar.gz').exists())

    def test_missing_array_does_not_replace_index(self):
        self.run_download()
        index = self.root / 'records_val.jsonl'
        before = index.read_bytes()
        (self.root / 'extracted/val/real/clean_hr512/val.fits').unlink()
        with self.assertRaises(FileNotFoundError):
            release.write_records(self.root, ['val'])
        self.assertEqual(index.read_bytes(), before)
        self.assertFalse(index.with_suffix('.jsonl.tmp').exists())

    def test_changed_public_index_is_rejected(self):
        with (self.root / 'dataset_index.jsonl').open('a') as f:
            f.write('{}\n')
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            release.public_records(self.root)

    def test_missing_license_stops_before_extraction(self):
        (self.root / 'LICENSE').unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_download()
        self.assertFalse((self.root / 'extracted').exists())

    def test_modified_source_credit_stops_before_extraction(self):
        (self.root / 'DATA_SOURCES.md').write_text('Changed source credit')
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            self.run_download()
        self.assertFalse((self.root / 'extracted').exists())

    def test_download_fetches_data_and_notice_revisions_separately(self):
        calls = []
        module = types.ModuleType('huggingface_hub')
        def snapshot(*args, **kwargs):
            calls.append(kwargs)
            return str(self.root)
        module.snapshot_download = snapshot
        with patch.dict(sys.modules, {'huggingface_hub': module}), \
             patch.object(sys, 'argv', ['prepare_dataset.py', '--local-dir', str(self.root),
                                       '--splits', 'val', '--download-only']), \
             contextlib.redirect_stdout(io.StringIO()):
            download.main()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]['revision'], release.REVISION)
        self.assertEqual(calls[1]['revision'], release.LICENSE_REVISION)
        self.assertEqual(set(calls[1]['allow_patterns']), set(release.LICENSE_FILES))


if __name__ == '__main__':
    unittest.main()
