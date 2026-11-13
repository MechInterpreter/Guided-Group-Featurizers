import json
from pathlib import Path
import pytest
from report_resume import cache_identity

def save(root,name,sha,**kw):
    p=root/'analysis/results'/name/'protocol.json'
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(dict(version=3,semantic_protocol='five native methods; fixture',source_sha=sha,**kw)))

def test_report_repair_keeps_prior_cache_source_salt(tmp_path):
    prior='2c115474fe6c54c9'+'a'*48
    save(tmp_path,'existing',prior)
    assert cache_identity(tmp_path,'b'*64)==prior

def test_new_run_does_not_alias_unknown_source(tmp_path):
    save(tmp_path,'unknown','c'*64)
    assert cache_identity(tmp_path,'b'*64)=='b'*64

def test_invalid_source_checksum_is_rejected(tmp_path):
    with pytest.raises(ValueError):cache_identity(tmp_path,'invalid')

def test_preserve_current_run_when_both_compatible_revisions_exist(tmp_path):
    prior='2c115474fe6c54c9'+'a'*48
    save(tmp_path,'first',prior)
    save(tmp_path,'later','f1670ac0adee1773'+'d'*48)
    assert cache_identity(tmp_path,'b'*64)==prior

def test_ambiguous_prefix_is_rejected(tmp_path):
    save(tmp_path,'first','2c115474fe6c54c9'+'a'*48)
    save(tmp_path,'other','2c115474fe6c54c9'+'d'*48)
    with pytest.raises(ValueError):cache_identity(tmp_path,'b'*64)
