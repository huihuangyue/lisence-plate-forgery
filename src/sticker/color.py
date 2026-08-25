"""颜色差异与背景残差。"""

from __future__ import annotations

import math

import numpy as np


def delta_e_ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """计算两个 CIE Lab 颜色的 CIEDE2000 色差（kL=kC=kH=1）。"""

    l1, a1, b1 = (float(value) for value in np.asarray(lab1).reshape(3))
    l2, a2, b2 = (float(value) for value in np.asarray(lab2).reshape(3))
    c1 = math.hypot(a1, b1)
    c2 = math.hypot(a2, b2)
    c_bar = (c1 + c2) / 2.0
    g = 0.5 * (1.0 - math.sqrt(c_bar**7 / (c_bar**7 + 25.0**7)))
    a1p = (1.0 + g) * a1
    a2p = (1.0 + g) * a2
    c1p = math.hypot(a1p, b1)
    c2p = math.hypot(a2p, b2)

    def hue_degrees(a: float, b: float) -> float:
        value = math.degrees(math.atan2(b, a))
        return value + 360.0 if value < 0.0 else value

    h1p = hue_degrees(a1p, b1) if c1p else 0.0
    h2p = hue_degrees(a2p, b2) if c2p else 0.0
    delta_lp = l2 - l1
    delta_cp = c2p - c1p
    if c1p * c2p == 0.0:
        delta_hp_degrees = 0.0
    elif abs(h2p - h1p) <= 180.0:
        delta_hp_degrees = h2p - h1p
    elif h2p <= h1p:
        delta_hp_degrees = h2p - h1p + 360.0
    else:
        delta_hp_degrees = h2p - h1p - 360.0
    delta_hp = 2.0 * math.sqrt(c1p * c2p) * math.sin(math.radians(delta_hp_degrees / 2.0))

    l_bar = (l1 + l2) / 2.0
    c_bar_p = (c1p + c2p) / 2.0
    if c1p * c2p == 0.0:
        h_bar = h1p + h2p
    elif abs(h1p - h2p) <= 180.0:
        h_bar = (h1p + h2p) / 2.0
    elif h1p + h2p < 360.0:
        h_bar = (h1p + h2p + 360.0) / 2.0
    else:
        h_bar = (h1p + h2p - 360.0) / 2.0

    t = (
        1.0
        - 0.17 * math.cos(math.radians(h_bar - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * h_bar))
        + 0.32 * math.cos(math.radians(3.0 * h_bar + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * h_bar - 63.0))
    )
    delta_theta = 30.0 * math.exp(-((h_bar - 275.0) / 25.0) ** 2)
    r_c = 2.0 * math.sqrt(c_bar_p**7 / (c_bar_p**7 + 25.0**7))
    s_l = 1.0 + 0.015 * (l_bar - 50.0) ** 2 / math.sqrt(20.0 + (l_bar - 50.0) ** 2)
    s_c = 1.0 + 0.045 * c_bar_p
    s_h = 1.0 + 0.015 * c_bar_p * t
    r_t = -math.sin(math.radians(2.0 * delta_theta)) * r_c
    l_term = delta_lp / s_l
    c_term = delta_cp / s_c
    h_term = delta_hp / s_h
    return float(math.sqrt(l_term**2 + c_term**2 + h_term**2 + r_t * c_term * h_term))


def opencv_lab_to_cie(values: np.ndarray) -> np.ndarray:
    """把 OpenCV uint8 Lab 标度转换为标准 CIE Lab 标度。"""

    result = np.asarray(values, dtype=np.float32).copy()
    result[..., 0] *= 100.0 / 255.0
    result[..., 1:] -= 128.0
    return result
