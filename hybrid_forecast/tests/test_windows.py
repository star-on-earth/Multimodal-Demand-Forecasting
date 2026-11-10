import copy

import numpy as np

from hff.config import effective_config
from hff.windows import Prepared, split_origins
from .conftest import TINY


def test_split_has_no_target_overlap(full_bundle):
    cfg = effective_config(TINY, full_bundle.n_days, 3)
    tr, te, ts = split_origins(full_bundle.n_days, cfg)
    assert tr.max() + cfg.horizon < ts          # last train target is before hold-out
    assert te.min() + 1 == ts                   # first test target is the first hold-out day
    assert te.max() + cfg.horizon == full_bundle.n_days - 1


def test_inputs_do_not_see_the_future(full_bundle):
    """Changing any data after a forecast origin must not change that origin's inputs."""
    cfg = effective_config(TINY, full_bundle.n_days, 3)
    _, te, ts = split_origins(full_bundle.n_days, cfg)
    o = int(te[len(te) // 2])
    b2 = copy.deepcopy(full_bundle)
    rng = np.random.default_rng(0)
    b2.sales[:, o + 1:] = rng.uniform(0, 1000, b2.sales[:, o + 1:].shape)
    rv = b2.reviews
    late = rv["day_idx"] > o
    rv.loc[late, "text"] = "completely different late text broken terrible"
    for iset in b2.images.values():
        iset.arr[iset.day_idx > o] = 0.0
    p1, p2 = Prepared(full_bundle, cfg, ts - 1), Prepared(b2, cfg, ts - 1)
    s1, s2 = p1.samples([o]), p2.samples([o])
    for k in s1.inputs:
        np.testing.assert_allclose(s1.inputs[k], s2.inputs[k], err_msg=k)
    np.testing.assert_allclose(s1.flat, s2.flat)
    assert not np.allclose(s1.y, s2.y)          # but the targets do change


def test_training_stats_only_use_training_period(full_bundle):
    cfg = effective_config(TINY, full_bundle.n_days, 3)
    _, _, ts = split_origins(full_bundle.n_days, cfg)
    b2 = copy.deepcopy(full_bundle)
    b2.sales[:, ts:] *= 7.0
    np.testing.assert_allclose(Prepared(full_bundle, cfg, ts - 1).scale, Prepared(b2, cfg, ts - 1).scale)
