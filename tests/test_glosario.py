from glosario import AYUDA


def test_textos_validos():
    assert AYUDA
    for clave, texto in AYUDA.items():
        assert isinstance(texto, str) and len(texto) > 10, clave
        assert "$" not in texto, f"'{clave}': el signo de dinero activa LaTeX en Streamlit"
