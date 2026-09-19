"""Offline migration checks: corruption, extraction boundaries, and cached downloads."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from download_experiment import download
from restore_experiment import restore, sha256


class PortableArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def archive(self, name='experiment/metrics.jsonl', bad_file_hash=False, link=False):
        content = b'{"step":100000}\n'
        record = dict(path=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        if bad_file_hash:
            record['sha256'] = '0' * 64
        manifest = dict(files=[record], training_commit='test-commit', exact_training_resume=False)
        target = self.root / 'test.tar.gz'
        with tarfile.open(target, 'w:gz') as output:
            data = json.dumps(manifest).encode()
            member = tarfile.TarInfo('MANIFEST.json')
            member.size = len(data)
            output.addfile(member, io.BytesIO(data))
            member = tarfile.TarInfo(name)
            if link:
                member.type = tarfile.SYMTYPE
                member.linkname = '../outside'
                output.addfile(member)
            else:
                member.size = len(content)
                output.addfile(member, io.BytesIO(content))
        return target

    def test_round_trip_and_refuse_overwrite(self):
        archive = self.archive()
        dest = self.root / 'restored'
        result = restore(archive, sha256(archive), dest)
        self.assertEqual(result['verified_files'], 1)
        self.assertEqual((dest / 'experiment/metrics.jsonl').read_bytes(), b'{"step":100000}\n')
        with self.assertRaisesRegex(ValueError, 'new or empty'):
            restore(archive, sha256(archive), dest)

    def test_corrupt_archive_rejected_before_extract(self):
        archive = self.archive()
        checksum = sha256(archive)
        with archive.open('ab') as output:
            output.write(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            restore(archive, checksum, self.root / 'restored')
        self.assertFalse((self.root / 'restored').exists())

    def test_file_hash_failure_has_no_success_marker(self):
        archive = self.archive(bad_file_hash=True)
        with self.assertRaisesRegex(ValueError, 'Extracted file mismatch'):
            restore(archive, sha256(archive), self.root / 'restored')
        self.assertFalse((self.root / 'restored/RESTORE-VERIFIED.json').exists())

    def test_path_traversal_and_links_rejected(self):
        for name, link in [('../outside', False), ('/outside', False), ('a\\b', False), ('experiment/link', True)]:
            with self.subTest(name=name):
                archive = self.archive(name=name, link=link)
                with self.assertRaisesRegex(ValueError, 'Unsafe'):
                    restore(archive, sha256(archive), self.root / 'restored')
                self.assertFalse((self.root / 'outside').exists())

    def test_verified_cached_parts_need_no_network(self):
        directory = self.root / 'download'
        parts = directory / 'parts'
        parts.mkdir(parents=True)
        records = []
        for i, content in enumerate([b'first', b'second']):
            name = f'part{i}'
            (parts / name).write_bytes(content)
            records.append(dict(index=i, name=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(),
                                url='https://github.com/test/releases/download/v1/' + name))
        manifest = self.root / 'archive.json'
        manifest.write_text(json.dumps(dict(archive=dict(filename='test.tar.gz', bytes=11,
            sha256=hashlib.sha256(b'firstsecond').hexdigest()), parts=records)))
        with patch('urllib.request.urlopen', side_effect=AssertionError('Must reuse cache')):
            self.assertEqual(download(manifest, directory).read_bytes(), b'firstsecond')


if __name__ == '__main__':
    unittest.main()
