from importlib.metadata import version


def test_mlx_version_matches_official_magenta_lock() -> None:
    assert version("mlx") == "0.31.2"
