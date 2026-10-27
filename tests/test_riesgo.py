import numpy as np
import pandas as pd
import pytest
from scipy import stats

from core import riesgo


def test_normal_coincide_con_formula_cerrada():
    # media 0, sigma 1 %: VaR95 = 1.645 %, ES95 = 2.063 % (valores clásicos)
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 400_000)
    x = (x - x.mean()) / x.std(ddof=1) * 0.01  # media 0 y sigma 0.01 exactas
    var, es = riesgo.var_es_normal(x, 0.95)
    assert var == pytest.approx(0.01 * 1.6449, abs=2e-5)
    assert es == pytest.approx(0.01 * 2.0627, abs=2e-5)


def test_historico_cercano_al_normal_en_datos_normales():
    x = np.random.default_rng(1).normal(0.0004, 0.012, 300_000)
    vh, eh = riesgo.var_es_historico(x, 0.99)
    vn, en = riesgo.var_es_normal(x, 0.99)
    assert vh == pytest.approx(vn, rel=0.02)
    assert eh == pytest.approx(en, rel=0.03)


def test_es_siempre_mayor_que_var():
    x = np.random.default_rng(2).standard_t(4, 3000) * 0.01
    for f in (riesgo.var_es_historico, riesgo.var_es_normal, riesgo.var_es_t):
        v, e = f(x, 0.975)
        assert e > v > 0


def test_t_cerrada_coincide_con_simulacion():
    params = (5.0, 0.0002, 0.008)
    var, es = riesgo.var_es_t([0.0] * 30, 0.975, params=params)
    sim = stats.t.rvs(5.0, 0.0002, 0.008, size=2_000_000, random_state=3)
    q = np.quantile(sim, 0.025)
    assert var == pytest.approx(-q, rel=0.01)
    assert es == pytest.approx(-sim[sim <= q].mean(), rel=0.01)


def test_t_con_muchos_grados_de_libertad_tiende_a_normal():
    params = (5000.0, 0.0, 0.01)
    v_t, e_t = riesgo.var_es_t([0.0] * 30, 0.99, params=params)
    z = stats.norm.ppf(0.01)
    assert v_t == pytest.approx(-0.01 * z, rel=1e-3)
    assert e_t == pytest.approx(0.01 * stats.norm.pdf(z) / 0.01, rel=1e-3)


def test_ajustar_t_recupera_grados_de_libertad():
    x = stats.t.rvs(4.0, 0.0, 0.01, size=20_000, random_state=4)
    df, _, scale = riesgo.ajustar_t(x)
    assert 3.3 < df < 4.8
    assert scale == pytest.approx(0.01, rel=0.05)


def test_cornish_fisher_sin_asimetria_ni_curtosis_es_normal(monkeypatch):
    x = np.random.default_rng(5).normal(0, 0.01, 1000)
    monkeypatch.setattr(stats, "skew", lambda *a, **k: 0.0)
    monkeypatch.setattr(stats, "kurtosis", lambda *a, **k: 0.0)
    assert riesgo.var_cornish_fisher(x, 0.99) == pytest.approx(riesgo.var_es_normal(x, 0.99)[0])


def test_cornish_fisher_aumenta_var_con_colas_pesadas():
    x = stats.t.rvs(5, size=20_000, random_state=6) * 0.01
    assert riesgo.var_cornish_fisher(x, 0.99) > riesgo.var_es_normal(x, 0.99)[0]


def test_montecarlo_converge_a_t_cerrada():
    x = stats.t.rvs(5.0, 0.0, 0.01, size=5000, random_state=7)
    params = riesgo.ajustar_t(x)
    v_mc, e_mc = riesgo.var_es_montecarlo(x, 0.99, n_sim=1_000_000, params_t=params)
    v_t, e_t = riesgo.var_es_t(x, 0.99, params)
    assert v_mc == pytest.approx(v_t, rel=0.03)
    assert e_mc == pytest.approx(e_t, rel=0.04)


def test_tabla_estructura_y_escalado_de_horizonte():
    x = np.random.default_rng(8).standard_t(5, 2000) * 0.01
    t1 = riesgo.tabla_var_es(x, 0.95, 1, n_sim=20_000)
    t10 = riesgo.tabla_var_es(x, 0.95, 10, n_sim=20_000)
    assert list(t1["Método"]) == ["Histórico", "Normal (paramétrico)", "t de Student", "Cornish-Fisher", "Monte Carlo (t)"]
    assert (t10["VaR"] / t1["VaR"]).tolist() == pytest.approx([np.sqrt(10)] * 5)


def test_var_historico_movil_coincide_con_calculo_manual():
    x = pd.Series(np.random.default_rng(9).normal(0, 0.01, 400), index=pd.bdate_range("2021-01-01", periods=400))
    w = 100
    v = riesgo.var_historico_movil(x, 0.95, w)
    assert v.iloc[:w - 1].isna().all()
    assert v.iloc[-1] == pytest.approx(-np.quantile(x.iloc[-w:], 0.05))


def test_es_historico_movil_coincide_con_calculo_manual():
    x = pd.Series(np.random.default_rng(10).normal(0, 0.01, 400), index=pd.bdate_range("2021-01-01", periods=400))
    w = 100
    es = riesgo.es_historico_movil(x, 0.95, w)
    for i in (w - 1, 250, 399):
        ventana = x.iloc[i - w + 1 : i + 1].to_numpy()
        q = np.quantile(ventana, 0.05)
        assert es.iloc[i] == pytest.approx(-ventana[ventana <= q].mean())
    assert es.iloc[: w - 1].isna().all()


def test_var_ewma_y_movil_dan_series_alineadas():
    x = pd.Series(np.random.default_rng(11).normal(0, 0.01, 300), index=pd.bdate_range("2021-01-01", periods=300))
    for m in riesgo.METODOS_MOVILES:
        v = riesgo.var_movil(x, m, 0.99, ventana=60)
        assert v.index.equals(x.index)
        assert v.dropna().gt(0).all()


def test_alpha_invalido():
    with pytest.raises(ValueError):
        riesgo.var_es_historico(np.random.default_rng(0).normal(size=100), 1.5)
