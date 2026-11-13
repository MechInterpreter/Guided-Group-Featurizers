"""Exercise secret and private-file detection using synthetic local fixtures."""
from pathlib import Path
import gzip
import importlib.util
import io
import subprocess
import tempfile
import unittest
import zipfile

SPEC=importlib.util.spec_from_file_location('repository_check',Path(__file__).resolve().parents[1]/'check_repository.py')
CHECK=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(CHECK)


class RepositorySafetyTests(unittest.TestCase):
    def test_normal_code_and_hashes_are_allowed(self):
        result=CHECK.audit([('research/example.py',b"value = 75\n"),('demo/hash.json',b'{"sha256":"'+b'a'*64+b'"}')])
        self.assertTrue(result['passed'])

    def test_nested_credentials_are_detected_without_echoing_values(self):
        token=('s'+'k-proj-'+('fixture'*8)).encode()
        output=io.BytesIO()
        with zipfile.ZipFile(output,'w') as archive:
            archive.writestr('notes.json.gz',gzip.compress(b'{"value":"'+token+b'"}'))
        result=CHECK.audit([('fixture.zip',output.getvalue())])
        self.assertFalse(result['passed'])
        self.assertEqual(result['findings'][0]['kind'],'openai_openrouter_key')
        self.assertNotIn(token.decode(),str(result))

    def test_nested_private_paths_are_rejected(self):
        output=io.BytesIO()
        with zipfile.ZipFile(output,'w') as archive:archive.writestr('.env',b'VALUE=example')
        self.assertFalse(CHECK.audit([('fixture.zip',output.getvalue())])['passed'])

    def test_ignored_private_files_and_model_weights_are_rejected(self):
        for name in ['demo/.local/example.txt','demo/.dev.vars','demo/model.safetensors']:
            self.assertFalse(CHECK.audit([(name,b'fixture')])['passed'],name)

    def test_arbitrary_known_secret_is_detected(self):
        value=b'local-fixture-'+b'x'*70
        result=CHECK.audit([('ordinary.txt',value)],extra_secrets=[value])
        self.assertEqual(result['findings'][0]['kind'],'known_local_secret')

    def test_scan_reads_index_bytes_even_after_working_copy_is_cleaned(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            subprocess.run(['git','init','-q',str(root)],check=True)
            token=('s'+'k-proj-'+('fixture'*8)).encode()
            path=root/'example.txt';path.write_bytes(token)
            subprocess.run(['git','-C',str(root),'add','example.txt'],check=True)
            path.write_bytes(b'clean working copy')
            previous=CHECK.ROOT
            try:
                CHECK.ROOT=root
                self.assertFalse(CHECK.audit(CHECK.staged_files())['passed'])
            finally:CHECK.ROOT=previous


if __name__=='__main__':unittest.main()
