#!/usr/bin/env python3
"""Verify a checked-out or packaged runtime against its checksum lock, offline."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def validate(lock: dict, root: Path = None, archive: Path = None) -> int:
    entries = [file for package in lock['packages'] for file in package['files']]
    expected = {file['destination']: file for file in entries}
    if len(expected) != len(entries):
        raise ValueError('Duplicate runtime destinations in lock')
    for name in expected:
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError(f'Unsafe runtime destination: {name}')

    def check(actual, read, link):
        if set(actual) != set(expected):
            raise ValueError(f'Runtime payload differs from lock: missing={sorted(set(expected)-set(actual))}, '
                             f'unexpected={sorted(set(actual)-set(expected))}')
        for name, file in expected.items():
            target = link(name)
            if 'symlink' in file:
                if target != file['symlink']:
                    raise ValueError(f'Symlink mismatch: {name}')
            elif target is not None or hashlib.sha256(read(name)).hexdigest() != file['sha256']:
                raise ValueError(f'Checksum mismatch: {name}')

    if root is not None:
        paths = {p.relative_to(root).as_posix(): p for p in root.rglob('*')
                 if (p.is_file() or p.is_symlink()) and p.suffix != '.meta'}
        check(paths, lambda name: paths[name].read_bytes(),
              lambda name: str(paths[name].readlink()) if paths[name].is_symlink() else None)
    else:
        with tarfile.open(archive, 'r:gz') as package:
            prefix = 'package/Runtime/Plugins/'
            members = [m for m in package.getmembers() if m.name.startswith(prefix)
                       and not m.isdir() and not m.name.endswith('.meta')]
            paths = {m.name[len(prefix):]: m for m in members}
            if len(paths) != len(members):
                raise ValueError('Duplicate runtime entries in archive')
            check(paths, lambda name: package.extractfile(paths[name]).read(),
                  lambda name: paths[name].linkname if paths[name].issym() or paths[name].islnk() else None)
    return len(expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--directory", type=Path)
    source.add_argument("--archive", type=Path)
    arguments = parser.parse_args()
    lock = json.loads(arguments.lock.read_text(encoding="utf-8"))
    count = validate(lock, arguments.directory, arguments.archive)
    print(f"Validated {count} runtime files and their checksums.")


if __name__ == "__main__":
    main()
