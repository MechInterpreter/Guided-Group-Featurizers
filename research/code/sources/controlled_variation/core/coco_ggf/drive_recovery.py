"""Transport-only recovery for the existing width/balance run."""
import errno
import time
from pathlib import Path
from types import MethodType


def install_drive_recovery(store):
    """Keep local saves; retry remote slots and reduce large Drive writes."""
    if getattr(store, '_width_drive_recovery', False):
        return
    original_slot = store._save_slot
    original_save = store.save

    def retry_slot(self, base, key, staging):
        if Path(base) != self.root:
            return original_slot(base, key, staging)
        for attempt in range(4):
            try:
                self.check_mount()
                return original_slot(base, key, staging)
            except OSError as exc:
                # A full disk or permissions failure needs intervention.
                if exc.errno in (errno.ENOSPC, errno.EACCES, errno.EPERM, errno.EROFS) or attempt == 3:
                    raise
                print(f'Drive checkpoint retry {attempt + 1}/3: {key}: {exc}', flush=True)
                time.sleep(2 ** attempt)

    def save(self, key, value, drive=True):
        if drive and isinstance(value, dict) and 'updates' in value and 'target' in value:
            u = int(value['updates'])
            drive = u % 250 == 0 or u >= int(value['target'])
        return original_save(key, value, drive=drive)

    store._save_slot = MethodType(retry_slot, store)
    store.save = MethodType(save, store)
    store._width_drive_recovery_original_save = original_save
    store._width_drive_recovery = True
    store.write('drive_recovery.json', dict(
        version=1, local_checkpoint_updates=50, drive_checkpoint_updates=250,
        completed_phases_saved_to_drive=True, remote_attempts=4,
        numerical_settings_changed=False, source_archive_unchanged=True,
        note='Drive transport patch; local two-slot checkpoints and SHA read-back remain intact.'))


# Apply only to this run's writable store, never to the frozen source stores.
if 'CTX' in globals():
    install_drive_recovery(CTX['store'])
