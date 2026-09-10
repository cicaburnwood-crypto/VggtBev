"""Verify the archive without importing models or touching GPUs."""
import ast
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
    metadata = json.loads((root / 'DEPLOYMENT_METADATA.json').read_text())
    expected = manifest['files_sha256']
    actual_paths = {p.relative_to(root).as_posix() for p in (root / 'snapshot').rglob('*')
                    if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    if actual_paths != set(expected):
        raise RuntimeError('Snapshot file list differs from manifest')
    for name, digest in expected.items():
        path = root / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError('Snapshot mismatch: ' + name)
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=name)
    code = root / 'snapshot/baseline_verifier/procthor_20k_benchmark/shmcamera2500_20260909_v6'
    for name, digest in metadata['deployed_core_sha256'].items():
        if hashlib.sha256((code / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Core differs from deployed acceptance hash: ' + name)
    protocol = json.loads((code / 'protocol.json').read_text())
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode()).hexdigest()
    if digest != metadata['protocol_sha256']:
        raise RuntimeError('Protocol differs from deployed version')
    print(json.dumps({'snapshot_files': len(expected),
                      'deployed_core_files': len(metadata['deployed_core_sha256']),
                      'protocol_id': protocol['protocol_id'], 'verified': True}, indent=2))


if __name__ == '__main__':
    main()
