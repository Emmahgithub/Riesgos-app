"""VaR y Expected Shortfall (ES).

Convenciones
------------
* ``alpha`` es el nivel de confianza (0.95, 0.99...). La probabilidad de cola es ``p = 1 - alpha``.
* VaR y ES se devuelven como **pérdida positiva**, en fracción de la posición y a 1 día.
  Para otro horizonte se usa la regla de la raíz del tiempo (``escalar_horizonte``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats

from .volatilidad import vol_ewma

METODOS_MOVILES = ("Histórico", "Normal", "EWMA")


# --------------------------------------------------------------------------- utilidades
def _cola(alpha: float) -> float:
    if not 0.5 < alpha < 1.0:
        raise ValueError("alpha debe estar entre 0.5 y 1 (por ejemplo 0.95)")
    return 1.0 - alpha


def _limpiar(r) -> np.ndarray:
    x = np.asarray(r, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 30:
        raise ValueError("Se requieren al menos 30 observaciones.")
    return x


def escalar_horizonte(valor: float, horizonte: int) -> float:
    """Regla de la raíz del tiempo: supone rendimientos i.i.d.; es una aproximación."""
    return float(valor * np.sqrt(horizonte))


# --------------------------------------------------------------------------- puntuales
def var_es_historico(r, alpha: float) -> tuple[float, float]:
    x = _limpiar(r)
    q = np.quantile(x, _cola(alpha))
    return float(-q), float(-x[x <= q].mean())


def var_es_normal(r, alpha: float) -> tuple[float, float]:
    x = _limpiar(r)
    p = _cola(alpha)
    mu, s = x.mean(), x.std(ddof=1)
    z = stats.norm.ppf(p)
    return float(-(mu + s * z)), float(-mu + s * stats.norm.pdf(z) / p)


def ajustar_t(r) -> tuple[float, float, float]:
    """Ajusta una t de Student por máxima verosimilitud: (gl, loc, escala).

    Los grados de libertad se acotan a >= 2.05 para que la varianza y el ES existan.
    """
    x = _limpiar(r)
    df, loc, scale = stats.t.fit(x)
    return float(max(df, 2.05)), float(loc), float(scale)


def var_es_t(r, alpha: float, params: tuple[float, float, float] | None = None) -> tuple[float, float]:
    """VaR y ES con t de Student ajustada (ES en forma cerrada)."""
    p = _cola(alpha)
    df, loc, scale = params if params is not None else ajustar_t(r)
    tq = stats.t.ppf(p, df)
    var = -(loc + scale * tq)
    es = -loc + scale * stats.t.pdf(tq, df) / p * (df + tq**2) / (df - 1.0)
    return float(var), float(es)


def var_cornish_fisher(r, alpha: float) -> float:
    """VaR con expansión de Cornish-Fisher (corrige la normal por asimetría y curtosis).

    Es una aproximación válida para asimetría y curtosis moderadas; con colas muy
    pesadas el cuantil ajustado puede dejar de ser monótono.
    """
    x = _limpiar(r)
    p = _cola(alpha)
    mu, s = x.mean(), x.std(ddof=1)
    sk = stats.skew(x, bias=False)
    ku = stats.kurtosis(x, fisher=True, bias=False)
    z = stats.norm.ppf(p)
    z_cf = z + (z**2 - 1) * sk / 6 + (z**3 - 3 * z) * ku / 24 - (2 * z**3 - 5 * z) * sk**2 / 36
    return float(-(mu + s * z_cf))


def var_es_montecarlo(
    r,
    alpha: float,
    n_sim: int = 100_000,
    dist: str = "t",
    seed: int | None = 42,
    params_t: tuple[float, float, float] | None = None,
) -> tuple[float, float]:
    """VaR y ES por simulación de una distribución ajustada ('t' o 'normal').

    Para un solo activo converge al resultado paramétrico; su utilidad real aparece
    en portafolios (simulación conjunta con la matriz de covarianzas).
    """
    x = _limpiar(r)
    rng = np.random.default_rng(seed)
    if dist == "t":
        df, loc, scale = params_t if params_t is not None else ajustar_t(x)
        sim = loc + scale * rng.standard_t(df, n_sim)
    elif dist == "normal":
        sim = rng.normal(x.mean(), x.std(ddof=1), n_sim)
    else:
        raise ValueError("dist debe ser 't' o 'normal'")
    q = np.quantile(sim, _cola(alpha))
    return float(-q), float(-sim[sim <= q].mean())


def tabla_var_es(r, alpha: float, horizonte: int = 1, n_sim: int = 100_000, seed: int | None = 42) -> pd.DataFrame:
    """Compara todos los métodos. Columnas: Método, VaR, ES, Notas (fracciones, ya escaladas)."""
    x = _limpiar(r)
    params = ajustar_t(x)
    filas = []

    v, e = var_es_historico(x, alpha)
    filas.append(("Histórico", v, e, ""))
    v, e = var_es_normal(x, alpha)
    filas.append(("Normal (paramétrico)", v, e, ""))
    v, e = var_es_t(x, alpha, params)
    filas.append(("t de Student", v, e, f"gl = {params[0]:.1f}"))
    filas.append(("Cornish-Fisher", var_cornish_fisher(x, alpha), np.nan, "ES no disponible"))
    v, e = var_es_montecarlo(x, alpha, n_sim, "t", seed, params)
    filas.append(("Monte Carlo (t)", v, e, f"{n_sim:,} simulaciones"))

    tabla = pd.DataFrame(filas, columns=["Método", "VaR", "ES", "Notas"])
    f = np.sqrt(horizonte)
    tabla[["VaR", "ES"]] = tabla[["VaR", "ES"]] * f
    return tabla


# --------------------------------------------------------------------------- móviles
def var_historico_movil(r: pd.Series, alpha: float, ventana: int) -> pd.Series:
    return -r.rolling(ventana).quantile(_cola(alpha))


def es_historico_movil(r: pd.Series, alpha: float, ventana: int) -> pd.Series:
    """ES histórico móvil, vectorizado con ventanas deslizantes (sin bucles)."""
    if ventana > len(r):
        raise ValueError("La ventana es mayor que la serie.")
    w = sliding_window_view(r.to_numpy(dtype=float), ventana)
    q = np.quantile(w, _cola(alpha), axis=1)
    mascara = w <= q[:, None]
    es = -(np.where(mascara, w, 0.0).sum(axis=1) / mascara.sum(axis=1))
    return pd.Series(es, index=r.index[ventana - 1 :]).reindex(r.index)


def var_normal_movil(r: pd.Series, alpha: float, ventana: int) -> pd.Series:
    z = stats.norm.ppf(_cola(alpha))
    return -(r.rolling(ventana).mean() + r.rolling(ventana).std() * z)


def var_ewma(r: pd.Series, alpha: float, lam: float = 0.94, calentamiento: int = 30) -> pd.Series:
    """VaR normal con volatilidad EWMA y media cero (convención RiskMetrics)."""
    z = stats.norm.ppf(_cola(alpha))
    v = -z * vol_ewma(r, lam)
    v.iloc[:calentamiento] = np.nan  # la EWMA tarda en converger
    return v


def var_movil(r: pd.Series, metodo: str, alpha: float, ventana: int = 252, lam: float = 0.94) -> pd.Series:
    """VaR de 1 día para cada fecha, calculado con información hasta esa fecha."""
    if metodo == "Histórico":
        return var_historico_movil(r, alpha, ventana)
    if metodo == "Normal":
        return var_normal_movil(r, alpha, ventana)
    if metodo == "EWMA":
        return var_ewma(r, alpha, lam)
    raise ValueError(f"Método desconocido: {metodo}")
