"""Start the local authenticated inference service."""
from pathlib import Path
import os
import secrets
import sys
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
token_path = ROOT / '.local' / 'inference-token'
token_path.parent.mkdir(exist_ok=True)
if not token_path.exists():
    token_path.write_text(secrets.token_urlsafe(48), encoding='utf-8')
os.environ['GGF_INFERENCE_TOKEN'] = token_path.read_text(encoding='utf-8').strip()
sys.path.insert(0, str(ROOT / 'server'))
uvicorn.run('inference:app', host='127.0.0.1', port=8765, access_log=False, limit_concurrency=8)
