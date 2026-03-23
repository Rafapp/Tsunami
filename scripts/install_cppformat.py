#!/usr/bin/env python3
"""
install_cppformat.py: installs the pre-push git hook to auto format code.
Should run after build.sh / build.bat, if not installed: python scripts/install_cppformat.py
"""
 

import sys
import stat
import subprocess
from pathlib import Path


def get_git_root():
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if result.returncode != 0:
        print("ERROR: not inside a git repository.")
        sys.exit(1)
    return Path(result.stdout.strip())


def main():
    repo_root = get_git_root()
    hook_path = repo_root / ".git" / "hooks" / "pre-push"
    python = "python" if sys.platform == "win32" else "python3"
    hook_content = f"""\
#!/usr/bin/env sh
exec {python} "$(git rev-parse --show-toplevel)/scripts/cppformat.py" "$@"
"""
    hook_path.write_text(hook_content)
 
    # Mark executable on Mac/Linux (no-op on Windows)
    if sys.platform != "win32":
        hook_path.chmod(hook_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
 
    print(f"  installed cppformat hook -> {hook_path}")
    print(f"  cppformat will now run before every git push\n")
 

if __name__ == "__main__":
    main()