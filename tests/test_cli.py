from mrt_local.cli import build_parser
from mrt_local.config import project_root
from mrt_local import download as download_module
from mrt_local.download import build_parser as build_download_parser


def test_cli_models() -> None:
    parser = build_parser()
    info = parser.parse_args(["info", "--model", "mrt2_small"])
    assert info.model == "mrt2_small"
    assert info.temperature == 1.3
    assert info.top_k == 40
    assert info.cfg_musiccoca == 3.0
    assert info.warmup_steps == 5
    assert info.seed == 0
    assert info.use_mapper is True
    assert info.pool_across_time is True
    serve = parser.parse_args(["serve", "--model", "mrt2_base"])
    assert serve.model == "mrt2_base"
    assert serve.port == 8765
    assert parser.parse_args(["serve", "--port", "9000"]).port == 9000
    generate = parser.parse_args(
        ["generate", "--prompt", "x", "--output", "x.mp3", "--format", "mp3"]
    )
    assert generate.format == "mp3"
    assert generate.bitrate is None
    audio_generate = parser.parse_args(
        ["generate", "--reference-audio", "reference.wav"]
    )
    assert str(audio_generate.reference_audio) == "reference.wav"
    advanced = parser.parse_args([
        "generate",
        "--weighted-prompt", "1", "ambient pads",
        "--weighted-prompt", "2.5", "powerful drums",
    ])
    assert advanced.weighted_prompt == [
        ["1", "ambient pads"], ["2.5", "powerful drums"]
    ]
    assert parser.parse_args(["generate", "--prompt", "x", "--drumless"]).drumless is True


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


def test_cli_help_is_descriptive(capsys) -> None:
    parser = build_parser()
    try:
        parser.parse_args(["generate", "-h"])
    except SystemExit as exc:
        assert exc.code == 0
    output = capsys.readouterr().out
    assert "--prompt TEXT" in output
    assert "--weighted-prompt WEIGHT TEXT" in output
    assert "--drumless" in output
    assert "--reference-audio PATH" in output
    assert "--midi PATH" in output
    assert "mrt2_small,mrt2_base" in output
    assert "--temperature FLOAT" in output
    assert "--format {wav,mp3}" in output
    assert "--bitrate KBPS" in output
    assert "--no-use-mapper" in output
    assert "遵循 MIDI 鼓点控制" in output
    assert "不保证整段音频可复现" in output
    assert "uv run mrt-local generate" in output


def test_top_level_help_contains_commands_and_examples(capsys) -> None:
    parser = build_parser()
    try:
        parser.parse_args(["-h"])
    except SystemExit as exc:
        assert exc.code == 0
    output = capsys.readouterr().out
    assert "可用命令" in output
    assert "uv run mrt-local generate" in output
    assert "mrt-download -h" in output
