#!/usr/bin/env python3
"""
cppformat.py - runs clang-format on changed C++ files before a push
If any files get reformatted, they are auto-amended into the last commit and
the push continues without requiring manual intervention.

Install: python scripts/install_cppformat.py
Manual:  python scripts/cppformat.py
"""

import subprocess
import sys
from pathlib import Path


CPP_EXTENSIONS = {".cpp", ".cc", ".h", ".hpp"}


def find_clang_format():
    """Return the clang-format binary name if found on PATH, else None."""
    for name in ["clang-format", "clang-format-18", "clang-format-17", "clang-format-16"]:
        try:
            subprocess.run([name, "--version"], capture_output=True, check=True)
            return name
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return None


def get_git_root():
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if result.returncode != 0:
        print("ERROR: not inside a git repository.")
        sys.exit(1)
    return Path(result.stdout.strip())


def get_changed_files(repo_root):
    """Get C++ files changed since the last push. Falls back to staged files."""
    # Files in commits not yet on remote
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD@{u}..HEAD"],
        capture_output=True, text=True, cwd=repo_root
    )
    if result.returncode != 0:
        # No upstream set yet, fall back to staged files
        result = subprocess.run(
            ["git", "diff", "--name-only", "--cached"],
            capture_output=True, text=True, cwd=repo_root
        )

    files = []
    for line in result.stdout.splitlines():
        path = repo_root / line.strip()
        if path.suffix in CPP_EXTENSIONS and path.exists():
            files.append(path)
    return files


def format_file(clang_format, path):
    """Run clang-format -i on a file. Returns True if the file was modified."""
    before = path.read_bytes()
    subprocess.run([clang_format, "-i", str(path)], check=True)
    after = path.read_bytes()
    return before != after


def main():
    print("\n  tsunami > cppformat\n")

    repo_root = get_git_root()
    files = get_changed_files(repo_root)

    if not files:
        print("No C++ files changed, did not run cppformat. Have changes been committed yet?\n")
        return 0

    clang_format = find_clang_format()
    if not clang_format:
        print("ERROR: clang-format not found on PATH.")
        print("  macOS: brew install clang-format")
        print("  Windows: winget install LLVM.LLVM")
        print("  Linux: sudo apt install clang-format\n")
        return 1

    reformatted = []
    for f in files:
        if format_file(clang_format, f):
            reformatted.append(f.relative_to(repo_root))
            print(f"  reformatted: {f.relative_to(repo_root)}")
        else:
            print(f"  ok:          {f.relative_to(repo_root)}")

    print()

    if reformatted:
        rel_paths = [str(f) for f in reformatted]
        subprocess.run(["git", "add"] + rel_paths, check=True, cwd=repo_root)
        subprocess.run(
            ["git", "commit", "--amend", "--no-edit"],
            check=True, cwd=repo_root
        )
        print("  cppformat: reformatted files amended into last commit - push continuing.\n")
        return 0

    print("  All files clean - push allowed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
