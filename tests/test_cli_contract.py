from anima_rfpar.cli import build_parser


def test_cli_has_expected_commands() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "check-assets" in help_text
    assert "plan-benchmark" in help_text
    assert "attack" in help_text
