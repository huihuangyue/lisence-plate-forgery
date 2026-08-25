from __future__ import annotations

import numpy as np

from sticker.color import delta_e_ciede2000


def test_ciede2000_reference_pair() -> None:
    first = np.array([50.0, 2.6772, -79.7751])
    second = np.array([50.0, 0.0, -82.7485])
    assert abs(delta_e_ciede2000(first, second) - 2.0425) < 1e-4


def test_ciede2000_is_symmetric_and_zero_for_identity() -> None:
    first = np.array([62.0, -18.0, 31.0])
    second = np.array([59.0, -12.0, 23.0])
    assert delta_e_ciede2000(first, first) == 0.0
    assert abs(delta_e_ciede2000(first, second) - delta_e_ciede2000(second, first)) < 1e-10
