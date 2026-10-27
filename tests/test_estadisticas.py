import numpy as np
import pandas as pd
import pytest

from core import estadisticas as est


def _serie(x):
    return pd.Series(x, index=pd.bdate_range("2020-01-01", periods=len(x)))


def test_momentos_coinciden_con_numpy():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0005, 0.01, 2000)
    res = est.resumen_estadistico(_serie(x))
    assert res["media_diaria"] == pytest.approx(x.mean())
    assert res["varianza_diaria"] == pytest.approx(x.var(ddof=1))
    assert res["vol_anual"] == pytest.approx(x.std(ddof=1) * np.sqrt(252))
    assert res["varianza_anual"] == pytest.approx(x.var(ddof=1) * 252)


def test_normal_tiene_curtosis_exceso_cercana_a_cero_y_no_rechaza_jb():
    x = np.random.default_rng(1).normal(0, 0.01, 5000)
    res = est.resumen_estadistico(_serie(x))
    assert abs(res["curtosis_exceso"]) < 0.3
    assert abs(res["asimetria"]) < 0.15
    assert res["jb_pvalor"] > 0.01


def test_t_de_student_tiene_colas_pesadas_y_rechaza_normalidad():
    x = np.random.default_rng(2).standard_t(4, 5000) * 0.01
    res = est.resumen_estadistico(_serie(x))
    assert res["curtosis_exceso"] > 1.0
    assert res["jb_pvalor"] < 0.001


def test_drawdown():
    p = pd.Series([100.0, 120.0, 90.0, 110.0])
    dd = est.drawdown(p)
    assert dd.min() == pytest.approx(90 / 120 - 1)
    assert dd.iloc[0] == 0.0


def test_rendimientos_simple_y_log():
    p = pd.Series([100.0, 110.0, 99.0])
    assert est.rendimientos(p, "simple").tolist() == pytest.approx([0.10, -0.10])
    assert est.rendimientos(p, "log").tolist() == pytest.approx([np.log(1.1), np.log(0.9)])
    with pytest.raises(ValueError):
        est.rendimientos(p, "otro")


def test_pocas_observaciones():
    with pytest.raises(ValueError):
        est.resumen_estadistico(_serie(np.zeros(10)))
