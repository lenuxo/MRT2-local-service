from mrt_local.cli import build_parser


def test_cli_models() -> None:
    parser = build_parser()
    assert parser.parse_args(["info", "--model", "mrt2_small"]).model == "mrt2_small"
    assert parser.parse_args(["serve", "--model", "mrt2_base"]).model == "mrt2_base"
