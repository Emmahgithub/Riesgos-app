"""Textos de ayuda (botones «?») de la interfaz. Admiten Markdown.

Evitar el signo de dinero en estos textos: Streamlit lo interpreta como fórmula LaTeX.
"""

AYUDA: dict[str, str] = {
    # ------------------------------------------------------------ parámetros
    "ticker": (
        "Símbolo de la acción en Yahoo Finance. Ejemplos: `AAPL`, `MSFT`. "
        "Las acciones mexicanas llevan el sufijo `.MX` (`WALMEX.MX`) y los índices empiezan con `^` (`^GSPC`)."
    ),
    "periodo": (
        "Rango de fechas del historial de precios. Más historia da estimaciones más estables, "
        "pero mezcla épocas de mercado muy distintas; de 3 a 5 años es un buen punto de partida."
    ),
    "confianza": (
        "Probabilidad con la que la pérdida **no** superará el VaR. "
        "Con 95 % se espera superarlo 1 de cada 20 días; con 99 %, 1 de cada 100."
    ),
    "horizonte": (
        "Días a los que se proyecta la pérdida. Se escala con la regla de la raíz del tiempo "
        "(VaR a h días ≈ VaR a 1 día × √h), que supone rendimientos independientes entre días."
    ),
    "posicion": "Monto invertido en la acción. Solo sirve para convertir los porcentajes de VaR y ES a dinero.",
    "rf": (
        "Rendimiento de una inversión sin riesgo (por ejemplo CETES o T-Bills). "
        "Se usa en Sharpe y Sortino. Déjalo en 0 si no lo necesitas."
    ),
    "tipo": (
        "**Simple**: `P_t / P_(t-1) − 1`, la ganancia o pérdida real de la posición. "
        "**Logarítmico**: `ln(P_t / P_(t-1))`, se suma en el tiempo y es útil en modelos. "
        "En días normales ambos son casi iguales."
    ),
    "ventana": (
        "Número de días recientes que usan los métodos móviles. Una ventana corta reacciona rápido "
        "pero es ruidosa; una larga es estable pero lenta. 252 equivale a un año bursátil."
    ),
    "lambda": (
        "Factor de decaimiento de la EWMA. Cerca de 1 el modelo olvida lento (curva suave); "
        "con valores menores pesan más los días recientes. RiskMetrics usa 0.94 con datos diarios."
    ),
    # ------------------------------------------------------------ métricas y estadísticos
    "precio": "Último precio de cierre, ajustado por dividendos y splits.",
    "n": "Número de rendimientos diarios usados en los cálculos.",
    "media_diaria": "Rendimiento promedio de un día.",
    "rend_anual": "Rendimiento diario promedio × 252 días bursátiles. Es un promedio histórico, no una predicción.",
    "varianza_diaria": "Varianza de los rendimientos diarios (desviación estándar al cuadrado).",
    "varianza": (
        "Varianza anualizada (varianza diaria × 252). Es la medida de riesgo que usa Markowitz; "
        "la volatilidad es su raíz cuadrada."
    ),
    "desv_diaria": "Desviación estándar de los rendimientos de un día.",
    "vol_anual": (
        "Desviación estándar de los rendimientos diarios × √252. "
        "Mide qué tanto se mueve el precio; es la medida de riesgo más usada."
    ),
    "asimetria": (
        "Indica hacia qué lado se inclina la distribución. **Negativa**: las pérdidas extremas son "
        "mayores que las ganancias extremas. En una normal vale 0."
    ),
    "curtosis": (
        "Curtosis en exceso: peso de las colas frente a una normal (que vale 0). **Positiva** significa "
        "que los eventos extremos son más frecuentes de lo que supone la normal, y el VaR normal subestima el riesgo."
    ),
    "minimo": "Peor rendimiento de un solo día en el periodo.",
    "maximo": "Mejor rendimiento de un solo día en el periodo.",
    "jb": (
        "Prueba Jarque-Bera: contrasta si la asimetría y la curtosis son compatibles con una normal. "
        "Con **p < 0.05** se rechaza normalidad."
    ),
    "jb_est": "Estadístico de la prueba Jarque-Bera. Valores altos indican alejamiento de la normal.",
    "jb_p": "Probabilidad de ver una desviación así si los datos fueran normales. Menor a 0.05: se rechaza normalidad.",
    "drawdown": (
        "Mayor caída porcentual desde un máximo hasta el mínimo posterior. "
        "Muestra la peor racha que habría sufrido quien tuviera la acción."
    ),
    "sharpe": (
        "(Rendimiento − tasa libre de riesgo) / volatilidad, anualizado. "
        "Rendimiento obtenido por cada unidad de riesgo; más alto es mejor."
    ),
    "sortino": "Parecido a Sharpe, pero solo penaliza la volatilidad a la baja (los días con pérdidas).",
    # ------------------------------------------------------------ gráficos
    "g_precio": "Precio de cierre ajustado a lo largo del periodo.",
    "g_dd": "Distancia porcentual entre el precio y su máximo previo en cada fecha. 0 % significa que está en máximos.",
    "g_hist": (
        "Barras: frecuencia real de los rendimientos diarios. **Normal**: curva con la misma media y desviación. "
        "**t**: curva de colas pesadas ajustada a los datos; *gl* son sus grados de libertad "
        "(menos gl, colas más pesadas; con 30 o más se parece a la normal)."
    ),
    "g_qq": (
        "Compara los cuantiles observados con los de una normal. Si los puntos siguen la recta, los datos son normales; "
        "si los extremos se curvan hacia afuera, hay colas pesadas."
    ),
    "g_vol": (
        "**Móvil 21 días**: desviación estándar de las últimas ~4 semanas. **EWMA**: promedio con pesos decrecientes, "
        "reacciona más rápido a los sobresaltos. La línea punteada es la volatilidad de todo el periodo."
    ),
    # ------------------------------------------------------------ VaR y ES
    "var_es": (
        "**VaR** (Valor en Riesgo): pérdida que no se superaría con la confianza elegida. "
        "**ES** (Expected Shortfall): pérdida promedio en los días en que sí se supera el VaR; "
        "describe qué tan mala es la cola."
    ),
    "var_hist": "Pérdida máxima esperada con la confianza elegida, estimada directo de los rendimientos pasados.",
    "es_hist": "Pérdida promedio en los peores días, aquellos que superan al VaR histórico.",
    "metodos": (
        "- **Histórico**: usa los cuantiles de los rendimientos pasados, sin suponer una distribución.\n"
        "- **Normal**: supone normalidad con la media y volatilidad estimadas; subestima colas pesadas.\n"
        "- **t de Student**: supone una t ajustada, que captura colas pesadas.\n"
        "- **Cornish-Fisher**: corrige el cuantil normal con la asimetría y la curtosis.\n"
        "- **Monte Carlo (t)**: simula miles de rendimientos de la t ajustada y mide su cola."
    ),
    "g_barras": (
        "Pérdida estimada por cada método. El ES siempre supera al VaR. "
        "Si los métodos difieren mucho, las colas de la distribución importan."
    ),
    # ------------------------------------------------------------ backtesting
    "bt_modelo": (
        "- **Histórico móvil**: cuantil de la ventana reciente.\n"
        "- **Normal móvil**: media y desviación de la ventana reciente.\n"
        "- **EWMA**: volatilidad EWMA con media cero.\n\n"
        "Cada modelo se compara contra los rendimientos reales."
    ),
    "bt_dias": "Días en los que hay un VaR pronosticado el día anterior para comparar con el rendimiento real.",
    "bt_viol": "Días en que la pérdida real superó al VaR pronosticado.",
    "bt_obs": "Violaciones entre días evaluados.",
    "bt_esp": "Es 1 − confianza. Un modelo bien calibrado debería fallar aproximadamente esta fracción de los días.",
    "g_bt": (
        "Línea azul: pérdida máxima esperada (−VaR) pronosticada el día anterior. "
        "Puntos rojos: días en que el rendimiento cayó por debajo de esa línea."
    ),
    "pruebas": (
        "Con un **p-valor menor a 0.05** se rechaza que el modelo esté bien calibrado. "
        "Cada prueba revisa un aspecto distinto."
    ),
    "kupiec": (
        "Prueba de cobertura: ¿la tasa observada de violaciones coincide con la esperada? "
        "Si se rechaza, el modelo subestima (o sobreestima) el riesgo."
    ),
    "christoffersen": (
        "Prueba de independencia: ¿las violaciones ocurren aisladas o en rachas? "
        "Si se rechaza, el modelo reacciona lento en periodos turbulentos."
    ),
    "cc": "Cobertura condicional: combina las dos pruebas anteriores (frecuencia e independencia).",
}
