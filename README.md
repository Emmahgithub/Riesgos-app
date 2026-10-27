# Riesgo de una acción

App web (también usable desde el celular) para medir el riesgo de una acción: estadísticas
descriptivas (media, varianza, asimetría, curtosis, normalidad), VaR y Expected Shortfall con
varios métodos, volatilidad y backtesting. Hecha con Python y Streamlit.

> Herramienta informativa y educativa. No constituye asesoría financiera.

## Qué calcula

| Pestaña | Contenido |
|---|---|
| Resumen | Precio, rendimiento y volatilidad anualizados, varianza, asimetría, curtosis en exceso, Sharpe, Sortino, drawdown, prueba Jarque-Bera |
| Distribución | Histograma vs. normal y t de Student ajustadas; gráfico Q-Q |
| VaR y ES | Histórico, normal, t de Student, Cornish-Fisher y Monte Carlo, con horizonte y monto de la posición |
| Volatilidad | Móvil (21 días) vs. EWMA (RiskMetrics) |
| Backtesting | VaR móvil (histórico, normal, EWMA) con pruebas de Kupiec y Christoffersen |

## Estructura

```
core/                  # cálculo puro, sin interfaz (reutilizable por una API o una PWA)
  datos.py             # descarga de precios (yfinance, aislada para poder cambiarla)
  estadisticas.py      # momentos, normalidad, Sharpe/Sortino, drawdown
  riesgo.py            # VaR/ES: histórico, normal, t, Cornish-Fisher, Monte Carlo, móviles
  volatilidad.py       # móvil y EWMA
  backtesting.py       # Kupiec, Christoffersen, cobertura condicional
tests/                 # pruebas contra fórmulas cerradas y simulaciones
streamlit_app.py       # interfaz
```

## Convenciones

- VaR y ES son **pérdidas positivas** como fracción de la posición, a 1 día; para otros
  horizontes se aplica la regla de la raíz del tiempo (aproximación).
- Curtosis en **exceso** (la normal vale 0). Varianza con `ddof=1`.
- Por defecto se usan rendimientos simples (corresponden al P&L real); hay opción logarítmica.
- En backtesting, el VaR de cada día se calcula con información hasta el día anterior.

## Uso local

```bash
git clone https://github.com/<tu-usuario>/<tu-repo>.git
cd <tu-repo>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run streamlit_app.py
```

Pruebas: `pytest -q`

## Despliegue

En [Streamlit Community Cloud](https://streamlit.io/cloud): conecta tu cuenta de GitHub,
elige este repositorio, rama `main` y archivo principal `streamlit_app.py`. La URL resultante
abre bien en el navegador del celular.

## Datos

Los precios vienen de Yahoo Finance mediante `yfinance`, una fuente no oficial que puede fallar
o limitar peticiones. Acciones mexicanas: sufijo `.MX` (p. ej. `WALMEX.MX`). Si `yfinance`
deja de funcionar, actualízalo (`pip install -U yfinance`) o sustituye `core/datos.py`.

## Hoja de ruta

- [ ] Módulo de portafolio: covarianza con shrinkage, Markowitz, frontera eficiente, contribución al riesgo
- [ ] VaR/ES de portafolio con Monte Carlo conjunto
- [ ] GARCH y pruebas de ES
- [ ] API (FastAPI) y PWA
