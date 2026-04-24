#!/usr/bin/env python3
"""lib/__init__.py의 __version__을 0.01 증가.

사용:
    python scripts/bump_version.py          # +0.01
    python scripts/bump_version.py --major  # +1.00 (대규모 변경)
    python scripts/bump_version.py --set 1.42  # 직접 지정

매 git commit 전에 실행하거나 pre-commit hook으로 연결하면 편리.
"""
import re
import sys
from pathlib import Path

INIT_PATH = Path(__file__).resolve().parent.parent / "lib" / "__init__.py"
VERSION_RE = re.compile(r'__version__\s*=\s*"v(\d+)\.(\d+)"')


def read_version(text):
    m = VERSION_RE.search(text)
    if not m:
        raise RuntimeError("__version__ 선언을 찾을 수 없습니다.")
    return int(m.group(1)), int(m.group(2))


def bump(text, *, major=False, explicit=None):
    maj, minor = read_version(text)
    if explicit is not None:
        new_maj, new_minor = explicit
    elif major:
        new_maj, new_minor = maj + 1, 0
    else:
        new_minor = minor + 1
        if new_minor >= 100:
            new_maj, new_minor = maj + 1, 0
        else:
            new_maj = maj
    new_text = VERSION_RE.sub(
        f'__version__ = "v{new_maj}.{new_minor:02d}"', text, count=1,
    )
    return new_text, f"v{maj}.{minor:02d}", f"v{new_maj}.{new_minor:02d}"


def main(argv):
    major = "--major" in argv
    explicit = None
    if "--set" in argv:
        idx = argv.index("--set")
        if idx + 1 >= len(argv):
            print("--set 뒤에 X.YY 형식 버전을 지정하세요. 예: --set 1.42")
            sys.exit(1)
        val = argv[idx + 1].lstrip("v")
        m = re.match(r"^(\d+)\.(\d+)$", val)
        if not m:
            print(f"잘못된 형식: {val}. 예: 1.42")
            sys.exit(1)
        explicit = (int(m.group(1)), int(m.group(2)))

    text = INIT_PATH.read_text(encoding="utf-8")
    new_text, old_v, new_v = bump(text, major=major, explicit=explicit)
    INIT_PATH.write_text(new_text, encoding="utf-8")
    print(f"{old_v} \u2192 {new_v}")


if __name__ == "__main__":
    main(sys.argv[1:])
