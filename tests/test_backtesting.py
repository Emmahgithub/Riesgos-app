import numpy as np
import pandas as pd
import pytest

from core import backtesting as bt


def test_kupiec_sin_desviacion_no_rechaza():
    lr, p = bt.kupiec_pof(x=5, n=500, p=0.01)  # exactamente lo esperado
    assert lr == pytest.approx(0.0, abs=1e-9)
    assert p == pytest.approx(1.0)


def test_kupiec_valor_conocido():
    # n=250, x=10, p=0.01 -> LR = -2[ln((.99^240 .01^10)/(.96^240 .04^10))]
    esperado = -2 * ((240 * np.log(0.99) + 10 * np.log(0.01)) - (240 * np.log(0.96) + 10 * np.log(0.04)))
    lr, p = bt.kupiec_pof(10, 250, 0.01)
    assert lr == pytest.approx(esperado)
    assert p < 0.001


def test_kupiec_cero_violaciones_no_da_nan():
    lr, p = bt.kupiec_pof(0, 250, 0.01)
    assert np.isfinite(lr) and 0 <= p <= 1


def test_christoffersen_detecta_agrupamiento():
    v = np.zeros(1000, dtype=bool)
    for k in range(10):
        v[50 + 100 * k : 53 + 100 * k] = True  # 10 rachas de 3 violaciones seguidas
    _, p = bt.christoffersen_independencia(v)
    assert p < 0.001


def test_christoffersen_no_rechaza_violaciones_espaciadas():
    v = np.zeros(1000, dtype=bool)
    v[::50] = True  # 20 violaciones aisladas
    _, p = bt.christoffersen_independencia(v)
    assert p > 0.05


def test_evaluar_iid_con_tasa_correcta_no_rechaza():
    v = np.random.default_rng(0).random(5000) < 0.05
    res = bt.evaluar(v, 0.95)
    assert res.n == 5000 and res.violaciones == int(v.sum())
    assert res.tasa_esperada == pytest.approx(0.05)
    assert res.p_pof > 0.01 and res.p_ind > 0.01 and res.p_cc > 0.01
    assert res.lr_cc == pytest.approx(res.lr_pof + res.lr_ind)


def test_evaluar_detecta_subestimacion():
    v = np.random.default_rng(1).random(2000) < 0.10  # el doble de lo esperado
    assert bt.evaluar(v, 0.95).p_pof < 0.001


def test_alinear_desplaza_el_var_un_dia():
    idx = pd.bdate_range("2022-01-03", periods=5)
    r = pd.Series([0.0, -0.05, 0.01, -0.02, 0.0], index=idx)
    var = pd.Series([0.03, 0.03, 0.03, 0.03, 0.03], index=idx)
    viol, previo = bt.alinear(r, var)
    assert len(viol) == 4  # el primer día no tiene pronóstico previo
    assert viol.tolist() == [True, False, False, False]  # solo -5 % < -3 %
    assert previo.iloc[0] == pytest.approx(0.03)
