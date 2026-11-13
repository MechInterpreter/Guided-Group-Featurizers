"""Reconstruct privacy-safe original ZIP bytes without bypassing original pins."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent / 'original_containers'

def archive_bytes(original_sha):
    index = json.loads((ROOT / 'index.json').read_bytes())
    if original_sha in index['withheld']:
        raise ValueError('This original ZIP contains private locators and is not redistributed. '
                         'Restore its anonymized source by original SHA with frozen_legacy/restore.py; '
                         'do not bypass original-container checksum checks.')
    record = index['archives'][original_sha]
    archive = ROOT / 'chunks.zip'
    if hashlib.sha256(archive.read_bytes()).hexdigest() != index['chunk_archive_sha256']:
        raise ValueError('Original ZIP chunk archive differs from its index.')
    parts = []
    with ZipFile(archive) as z:
        for item in record['chunks']:
            raw = z.read(item['sha256'])
            if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
                raise ValueError('Original ZIP chunk differs from its index.')
            parts.append(raw)
    result = b''.join(parts)
    if len(result) != record['bytes'] or hashlib.sha256(result).hexdigest() != original_sha:
        raise ValueError('Reconstructed ZIP differs from the original container identity.')
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original_sha', nargs='?')
    parser.add_argument('output', nargs='?', type=Path)
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    if args.list:
        index = json.loads((ROOT / 'index.json').read_bytes())
        for digest in index['archives']:
            print(digest, 'exact original available')
        for digest in index['withheld']:
            print(digest, 'anonymized source restoration only')
        return
    if args.original_sha is None or args.output is None:
        parser.error('Supply an original SHA and a new output ZIP path, or --list.')
    raw = archive_bytes(args.original_sha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        stream.write(raw)
    print('Verified and wrote', args.original_sha)

if __name__ == '__main__':
    main()
