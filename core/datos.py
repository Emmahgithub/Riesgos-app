"""Descarga de precios de mercado.

La fuente (yfinance) está aislada en este módulo para poder cambiarla por otra
(API de pago, Banxico, etc.) sin tocar el resto del proyecto.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


class ErrorDatos(Exception):
    """No se pudieron obtener precios válidos para el ticker pedido."""


def descargar_precios(ticker: str, inicio: date, fin: date, min_obs: int = 60) -> pd.Series:
    """Precios de cierre ajustados (dividendos y splits) de un ticker.

    Acciones mexicanas en Yahoo Finance llevan el sufijo ``.MX`` (p. ej. ``WALMEX.MX``).
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise ErrorDatos("Escribe un ticker.")
    if inicio >= fin:
        raise ErrorDatos("La fecha inicial debe ser anterior a la final.")

    try:
        import yfinance as yf  # import perezoso: el núcleo no lo necesita para calcular

        df = yf.download(
            ticker,
            start=inicio,
            end=fin + timedelta(days=1),  # 'end' es exclusivo en yfinance
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:  # red, límites de la API, etc.
        raise ErrorDatos(f"No se pudo descargar {ticker}: {exc}") from exc

    if df is None or df.empty or "Close" not in df:
        raise ErrorDatos(f"Sin datos para '{ticker}'. Revisa el ticker y el rango de fechas.")

    cierre = df["Close"]
    if isinstance(cierre, pd.DataFrame):  # versiones recientes devuelven columnas por ticker
        cierre = cierre.iloc[:, 0]
    cierre = cierre.dropna().astype(float)
    if getattr(cierre.index, "tz", None) is not None:
        cierre.index = cierre.index.tz_localize(None)

    if len(cierre) < min_obs:
        raise ErrorDatos(f"Solo hay {len(cierre)} observaciones para '{ticker}'; se necesitan al menos {min_obs}.")

    cierre.name = ticker
    return cierre
