# Security

Keep credentials out of commits and issue reports. Do not attach `.env`, `.dev.vars`, private checkpoints or uploaded images. Run `python scripts/check_repository.py --history` before a push and retain GitHub push protection.

The demo is intended for local use and binds to localhost. Its backend authenticates requests with a generated local token. Uploads are limited and decoded in memory. The recorded research artifacts are not a multi-user hosted service. Deployment requires its own authentication, capacity and privacy review.

Only load model checkpoints and joblib artifacts from trusted local sources.
