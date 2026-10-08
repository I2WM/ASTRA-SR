"""Download, verify and extract selected splits of the public release."""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

# GitHub: scripts/prepare_dataset.py; HF: prepare_dataset.py beside release_utils.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from release_utils import (ARTIFACTS, COUNTS, INDEX_SHA, PROTOCOL_SHA, REPO,
                           REVISION, SUMS_SHA, LICENSE_REVISION, LICENSE_FILES,
                           check_hash, contained_path, write_records)


def selected_archives(root, splits, sums):
    selected = sorted(name for name in sums if name.endswith('.tar.gz')
                      and name.split('-', 1)[0] in splits)
    if any(not any(name.startswith(s + '-') for name in selected) for s in splits):
        raise ValueError('Checksum manifest is missing a requested split')
    paths = [contained_path(root / 'archives', name) for name in selected]
    for path in paths:
        check_hash(path, sums[path.name])
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-dir', type=Path, default=Path('astrasr_data'))
    parser.add_argument('--splits', nargs='+', choices=COUNTS, default=['val', 'test'])
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--local-only', action='store_true', help='Verify/extract an existing download')
    parser.add_argument('--artifacts', action='store_true', help='Also download weights and the PSF bank')
    parser.add_argument('--remove-archives', action='store_true', help='Remove verified archives after extraction')
    parser.add_argument('--keep-archives', action='store_true', help='Archives are kept by default')
    args = parser.parse_args()
    if args.remove_archives and args.keep_archives:
        parser.error('--remove-archives conflicts with --keep-archives')
    if not hasattr(tarfile, 'data_filter'):
        parser.error('Use Python 3.12+ or a patched Python with tarfile.data_filter')
    root = args.local_dir.resolve()
    if not args.local_only:
        from huggingface_hub import snapshot_download
        patterns = ['SHA256SUMS.txt', 'dataset_index.jsonl', 'x2_dataset_protocol_lrdegrade_v2.json']
        patterns += [f'archives/{split}-*.tar.gz' for split in args.splits]
        if args.artifacts:
            patterns += list(ARTIFACTS)
        snapshot_download(REPO, repo_type='dataset', revision=REVISION,
                          local_dir=str(root), allow_patterns=patterns, max_workers=2)
        snapshot_download(REPO, repo_type='dataset', revision=LICENSE_REVISION,
                          local_dir=str(root), allow_patterns=list(LICENSE_FILES), max_workers=2)
    for name, digest in LICENSE_FILES.items():
        check_hash(root / name, digest)
    check_hash(root / 'SHA256SUMS.txt', SUMS_SHA)
    check_hash(root / 'dataset_index.jsonl', INDEX_SHA)
    check_hash(root / 'x2_dataset_protocol_lrdegrade_v2.json', PROTOCOL_SHA)
    sums = {}
    for line in (root / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        digest, name = line.split(None, 1)
        sums[name.strip()] = digest
    archives = selected_archives(root, set(args.splits), sums)
    if args.artifacts:
        for name, digest in ARTIFACTS.items():
            check_hash(root / name, digest)
    print(f'Verified {len(archives)} requested archives at revision {REVISION}', flush=True)
    if args.download_only:
        return
    output = root / 'extracted'
    output.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        print(f'Extracting {archive.name}', flush=True)
        with tarfile.open(archive, 'r:gz') as stream:
            stream.extractall(output, filter='data')
    records = write_records(root, args.splits)
    if args.remove_archives:
        for archive in archives:
            archive.unlink()
    print(json.dumps({'records': str(records), 'splits': sorted(set(args.splits)),
                      'note': 'Local index paths differ from the historical server index.'}))


if __name__ == '__main__':
    main()
