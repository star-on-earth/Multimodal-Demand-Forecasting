import os
import sys
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import pytest  # noqa: E402

from hff import ingest, synth  # noqa: E402
from hff.config import Config  # noqa: E402

TINY = Config(epochs=2, batch_size=128, seeds=(0,), test_days=90)


@pytest.fixture(scope="session")
def raw():
    return synth.generate(n_products=3, n_days=380, seed=5, signal=True)


@pytest.fixture(scope="session")
def full_bundle(raw):
    return ingest.build_bundle(raw.sales, raw.reviews, ingest.images_from_raw(raw, TINY.img_size), TINY)


@pytest.fixture(scope="session")
def sales_bundle(raw):
    return ingest.build_bundle(raw.sales, None, None, TINY)
