from __future__ import annotations

import importlib
import sys

PACKAGES = [
    'streamlit',
    'numpy',
    'pandas',
    'matplotlib',
    'scipy',
    'skimage',
    'joblib',
    'tifffile',
    'imageio',
    'PIL',
]


def main() -> int:
    print(f'Python executable: {sys.executable}')
    print(f'Python version   : {sys.version.split()[0]}')
    failed = False
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, '__version__', 'unknown')
            print(f'{name:12s} OK   {version}')
        except Exception as exc:  # pragma: no cover - diagnostic helper
            failed = True
            print(f'{name:12s} FAIL {exc}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
