import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from storage import Storage  # noqa: E402


@pytest.fixture
def tmp_storage(tmp_path):
    return Storage(tmp_path / "t.db")
