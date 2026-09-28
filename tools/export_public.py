#!/usr/bin/env python3
"""Build the allowlisted WaveAlign GitHub tree, without private experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "README.md", "LICENSE", "pyproject.toml", ".gitignore",
    "generate.py", "app.py", "tools/export_public.py",
    "examples/prompts.jsonl", ".github/workflows/tests.yml",
)
PACKAGE = tuple(sorted((ROOT / "wavealign").glob("*.py")))
TESTS = (ROOT / "tests/test_release.py",)


def export(destination: Path) -> list[str]:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    if destination == ROOT or ROOT in destination.parents:
        raise ValueError("export outside the development tree")
    sources = [ROOT / name for name in FILES] + list(PACKAGE) + list(TESTS)
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix in (".py", ".toml", ".md"):
            contents = source.read_text(encoding="utf-8")
            if "/mnt/" + "shared-storage-user/" in contents:
                raise ValueError(f"workspace path in release file: {source}")
    destination.mkdir(parents=True)
    copied = []
    for source in sources:
        relative = source.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(str(relative))
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    files = export(args.destination)
    print(f"Exported {len(files)} files to {args.destination.resolve()}")


if __name__ == "__main__":
    main()
