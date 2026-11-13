# Repository safety

This repository was assembled from an allowlist of the verified research release and demo sources/public evidence. Existing Git history, private research archives, runtime credentials, author-machine files, model weights, caches and uploaded images are excluded. The frozen research file manifest is retained under `research/FILES.json`.

Before committing or pushing:

```sh
git add .
python scripts/check_repository.py --history --report .local/repository-safety.json
git diff --cached --stat
```

The check examines the exact Git index bytes, nested ZIP/NPZ/GZIP contents, blocked private paths, known credential formats and oversized blobs. `--history` also checks reachable Git blob history. Findings identify file paths and line numbers without printing suspected credentials. Never bypass a finding merely because a file is ignored elsewhere.

Install local hooks once:

```sh
git config core.hooksPath .githooks
```

The pre-commit and pre-push hooks run the same check and stop on findings. GitHub push protection is an additional safeguard when publishing; enable it in repository settings. No scanner can guarantee the absence of every possible secret. Review staged content before publication, and rotate a credential if it was ever committed or disclosed. Removing it from the latest version alone does not remove it from history.

Never commit `.local/`, `.dev.vars`, `.env`, model weights, runtime tokens, personal uploads, private archive ZIPs, account configuration or the old working repositories' `.git/` directories. Original third-party credits and the preprint's intended public author information are legitimate attribution, not credentials.

The repository is scanned before committing and pushing. Publishing or changing repository visibility is a separate action from editing files. Keep the project private until the intended public release.
