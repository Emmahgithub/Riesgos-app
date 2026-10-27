"""Backtesting de VaR: Kupiec (cobertura) y Christoffersen (independencia)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy  # xlogy(0, 0) = 0: evita NaN con cero violaciones


@dataclass(frozen=True)
class ResultadoBacktest:
    n: int
    violaciones: int
    tasa_observada: float
    tasa_esperada: float
    lr_pof: float
    p_pof: float
    lr_ind: float
    p_ind: float
    lr_cc: float
    p_cc: float


def alinear(r: pd.Series, var_pronostico: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Empareja cada rendimiento con el VaR pronosticado el día anterior.

    ``var_pronostico[t]`` usa información hasta t, así que se desplaza un día.
    Devuelve (violaciones booleanas, var_previo) sobre las fechas válidas.
    """
    var_previo = var_pronostico.shift(1)
    valido = var_previo.notna() & r.notna()
    r, var_previo = r[valido], var_previo[valido]
    return (r < -var_previo), var_previo


def kupiec_pof(x: int, n: int, p: float) -> tuple[float, float]:
    """Proporción de fallas (POF): ¿la tasa de violaciones x/n coincide con p?"""
    if n == 0:
        return float("nan"), float("nan")
    pi = x / n
    ll0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll1 = xlogy(n - x, 1 - pi) + xlogy(x, pi)
    lr = max(-2.0 * (ll0 - ll1), 0.0)
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen_independencia(viol: np.ndarray) -> tuple[float, float]:
    """¿Las violaciones ocurren agrupadas (dependencia) o de forma independiente?"""
    v = np.asarray(viol, dtype=int)
    a, b = v[:-1], v[1:]
    n00 = int(np.sum((a == 0) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n11 = int(np.sum((a == 1) & (b == 1)))
    total = n00 + n01 + n10 + n11
    if total == 0:
        return float("nan"), float("nan")
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / total
    ll0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll1 = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr = max(-2.0 * (ll0 - ll1), 0.0)
    return float(lr), float(stats.chi2.sf(lr, 1))


def evaluar(viol, alpha: float) -> ResultadoBacktest:
    """Kupiec + Christoffersen + cobertura condicional (suma, chi-cuadrada con 2 gl)."""
    v = np.asarray(viol, dtype=bool)
    n, x, p = int(v.size), int(v.sum()), 1.0 - alpha
    lr_pof, p_pof = kupiec_pof(x, n, p)
    lr_ind, p_ind = christoffersen_independencia(v)
    lr_cc = lr_pof + lr_ind
    return ResultadoBacktest(
        n=n,
        violaciones=x,
        tasa_observada=x / n if n else float("nan"),
        tasa_esperada=p,
        lr_pof=lr_pof,
        p_pof=p_pof,
        lr_ind=lr_ind,
        p_ind=p_ind,
        lr_cc=lr_cc,
        p_cc=float(stats.chi2.sf(lr_cc, 2)),
    )
