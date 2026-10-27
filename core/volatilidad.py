"""Estimadores de volatilidad diaria."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .estadisticas import DIAS_ANIO


def vol_movil(r: pd.Series, ventana: int = 21) -> pd.Series:
    """Desviación estándar móvil (diaria)."""
    return r.rolling(ventana).std()


def vol_ewma(r: pd.Series, lam: float = 0.94) -> pd.Series:
    """Volatilidad EWMA (RiskMetrics): var_t = lam*var_{t-1} + (1-lam)*r_t².

    El valor en t usa información hasta t. Para pronosticar t+1 hay que desplazar
    la serie un día (ver ``backtesting.alinear``).
    """
    if not 0.0 < lam < 1.0:
        raise ValueError("lam debe estar entre 0 y 1")
    return np.sqrt((r**2).ewm(alpha=1.0 - lam, adjust=False).mean())


def anualizar(vol_diaria: pd.Series | float, dias: int = DIAS_ANIO):
    return vol_diaria * np.sqrt(dias)
