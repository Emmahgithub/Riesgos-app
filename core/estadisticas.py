"""Estadísticas descriptivas y de desempeño de una serie de rendimientos."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

DIAS_ANIO = 252


def rendimientos(precios: pd.Series, tipo: str = "simple") -> pd.Series:
    """Rendimientos diarios a partir de precios. ``tipo``: 'simple' o 'log'."""
    precios = precios.dropna()
    if tipo == "simple":
        r = precios.pct_change()
    elif tipo == "log":
        r = np.log(precios).diff()
    else:
        raise ValueError("tipo debe ser 'simple' o 'log'")
    return r.dropna()


def drawdown(precios: pd.Series) -> pd.Series:
    """Caída porcentual desde el máximo previo (valores <= 0)."""
    return precios / precios.cummax() - 1.0


def resumen_estadistico(r: pd.Series, rf_anual: float = 0.0, dias: int = DIAS_ANIO) -> dict[str, float]:
    """Momentos, normalidad y razones de desempeño de los rendimientos diarios.

    Varianza y desviación usan ddof=1. Asimetría y curtosis en exceso (0 para la
    normal) usan la corrección de muestra pequeña (bias=False).
    """
    x = r.dropna().to_numpy(dtype=float)
    n = x.size
    if n < 30:
        raise ValueError("Se requieren al menos 30 observaciones.")

    media = x.mean()
    varianza = x.var(ddof=1)
    desv = float(np.sqrt(varianza))

    exceso = x - rf_anual / dias
    sharpe = exceso.mean() / desv * np.sqrt(dias) if desv > 0 else np.nan
    semidesv = np.sqrt(np.mean(np.minimum(exceso, 0.0) ** 2))
    sortino = exceso.mean() / semidesv * np.sqrt(dias) if semidesv > 0 else np.nan

    jb = stats.jarque_bera(x)
    return {
        "n": float(n),
        "media_diaria": float(media),
        "media_anual": float(media * dias),
        "varianza_diaria": float(varianza),
        "varianza_anual": float(varianza * dias),
        "desv_diaria": desv,
        "vol_anual": float(desv * np.sqrt(dias)),
        "asimetria": float(stats.skew(x, bias=False)),
        "curtosis_exceso": float(stats.kurtosis(x, fisher=True, bias=False)),
        "minimo": float(x.min()),
        "maximo": float(x.max()),
        "jb_estadistico": float(jb.statistic),
        "jb_pvalor": float(jb.pvalue),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
    }
