import sys
from pathlib import Path

# ensure top-level project directory is on the import path for tests
ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
