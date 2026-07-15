"""Pytest bootstrap for ideal-cnt-cb-network.

Makes the in-repo ``cntcb`` package importable when the test suite is run
from a bare checkout (``pytest -q`` without ``pip install -e .``).  When the
package *is* installed (as in CI), the repo root simply shadows the editable
install pointing at the same source tree, so this is a no-op in practice.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
