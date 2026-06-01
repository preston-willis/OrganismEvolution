import glob
import os

import pytest

_REAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture(scope="session", autouse=True)
def isolate_quantum_checkpoint_data(tmp_path_factory):
    import quantum.checkpoint as cp

    before = set(glob.glob(os.path.join(_REAL_DATA_DIR, "quantum_gen*.pt")))

    test_dir = tmp_path_factory.mktemp("quantum_checkpoint_data")
    original = cp.DATA_DIR
    cp.DATA_DIR = str(test_dir)
    yield
    cp.DATA_DIR = original

    after = set(glob.glob(os.path.join(_REAL_DATA_DIR, "quantum_gen*.pt")))
    for path in after - before:
        os.remove(path)
