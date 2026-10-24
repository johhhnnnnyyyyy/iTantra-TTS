from itantra_tts import __version__


def test_pre_alpha_version_is_exposed() -> None:
    assert __version__ == "0.1.0a0"
