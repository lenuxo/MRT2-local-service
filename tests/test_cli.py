from mrt_local.cli import build_parser
from mrt_local.config import project_root
from mrt_local import download as download_module
from mrt_local.download import build_parser as build_download_parser


def test_cli_models() -> None:
    parser = build_parser()
    assert parser.parse_args(["info", "--model", "mrt2_small"]).model == "mrt2_small"
    assert parser.parse_args(["serve", "--model", "mrt2_base"]).model == "mrt2_base"


def test_download_defaults_to_project_models_directory() -> None:
    args = build_download_parser().parse_args([])
    assert args.models == []
    assert args.model_root == project_root() / "models"


def test_download_command_defaults_to_small(monkeypatch) -> None:
    called = {}

    def fake_download(models, model_root) -> None:
        called["models"] = models
        called["model_root"] = model_root

    monkeypatch.setattr(download_module, "download", fake_download)
    download_module.main([])
    assert called == {
        "models": ["mrt2_small"],
        "model_root": project_root() / "models",
    }
