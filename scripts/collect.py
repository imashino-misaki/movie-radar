"""Run the collector: python scripts/collect.py [--data-dir data]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collector.run import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
