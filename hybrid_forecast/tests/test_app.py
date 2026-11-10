import io
import os

import pytest
from streamlit.testing.v1 import AppTest

from hff import synth

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setenv("HFF_EPOCHS", "2")


def _new():
    at = AppTest.from_file(APP, default_timeout=900)
    at.run()
    return at


def test_starts_with_guidance_not_errors():
    at = _new()
    assert not at.exception and at.info


def test_synthetic_end_to_end():
    at = _new()
    at.sidebar.slider[0].set_value(2)       # products
    at.sidebar.slider[1].set_value(450)     # days
    at.sidebar.button[0].click().run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    assert len(at.tabs) == 3 and len(at.dataframe) >= 2
    # forecast tab
    fb = [b for b in at.button if "forecast the next" in b.label][0]
    fb.click().run()
    assert not at.exception, at.exception
    assert any("Why this forecast" in s.value for s in at.subheader)


def test_upload_sales_only_runs():
    raw = synth.generate(n_products=2, n_days=450, seed=9)
    at = _new()
    at.sidebar.radio[0].set_value("Upload my own").run()
    csv = raw.sales.to_csv(index=False).encode()
    at.sidebar.file_uploader[0].upload("sales.csv", csv, "text/csv")
    at.sidebar.run()
    at.sidebar.button[0].click().run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    status = at.dataframe[0].value
    assert list(status["usable"]) == ["yes", "no", "no"]      # sales only; the rest reported, not fatal


def test_upload_nothing_is_a_clear_error_not_a_crash():
    at = _new()
    at.sidebar.radio[0].set_value("Upload my own").run()
    at.sidebar.button[0].click().run()
    assert not at.exception, at.exception
    assert at.error                      # a readable message, not a traceback
