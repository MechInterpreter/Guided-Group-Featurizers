import errno
from pathlib import Path
import pytest
from coco_baselines.storage import Store
from coco_ggf import drive_recovery


def test_remote_retry_local_resume_final_flush_and_failure(tmp_path,monkeypatch):
    s=Store(tmp_path/'out',dict(test=1),tmp_path/'local',require_drive=False)
    original=s._save_slot;attempts=[]
    def flaky(base,key,staging):
        if base==s.root:
            attempts.append(key)
            if len(attempts)<3:raise FileNotFoundError(errno.ENOENT,'temporary')
        return original(base,key,staging)
    s._save_slot=flaky
    drive_recovery.install_drive_recovery(s);drive_recovery.install_drive_recovery(s)
    monkeypatch.setattr(drive_recovery.time,'sleep',lambda _:None)
    s.save('phase',dict(updates=250,target=500))
    assert len(attempts)==3
    s.save('phase',dict(updates=300,target=500))
    assert s.load('phase')['updates']==300 and len(attempts)==3
    remote=Store(s.output,s.config,tmp_path/'empty',require_drive=False)
    assert remote.load('phase')['updates']==250
    s.save('phase',dict(updates=500,target=500))
    assert remote.load('phase')['updates']==500
    b=Store(tmp_path/'bad',dict(test=2),tmp_path/'bad_local',require_drive=False)
    original_bad=b._save_slot;calls=[]
    def unavailable(base,key,staging):
        if base==b.root:
            calls.append(key);raise FileNotFoundError(errno.ENOENT,'unavailable')
        return original_bad(base,key,staging)
    b._save_slot=unavailable;drive_recovery.install_drive_recovery(b)
    with pytest.raises(FileNotFoundError):b.save('phase',dict(updates=250,target=500))
    assert len(calls)==4 and b.load('phase')['updates']==250


def test_permanent_remote_failure_is_not_retried(tmp_path,monkeypatch):
    s=Store(tmp_path/'out',dict(test=3),tmp_path/'local',require_drive=False)
    original=s._save_slot;calls=[]
    def full(base,key,staging):
        if base==s.root:
            calls.append(key);raise OSError(errno.ENOSPC,'disk full')
        return original(base,key,staging)
    s._save_slot=full;drive_recovery.install_drive_recovery(s)
    with pytest.raises(OSError):s.save('phase',dict(updates=250,target=500))
    assert len(calls)==1 and s.load('phase')['updates']==250
