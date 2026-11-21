import json

from absa.cli import main


def test_cli_json_output(capsys):
    assert main(["analyze", "The food was great but the service was slow.", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert {r["aspect"]: r["sentiment"] for r in out} == {"food": "positive", "service": "negative"}


def test_cli_table_output(capsys):
    assert main(["analyze", "The staff are friendly."]) == 0
    assert "staff" in capsys.readouterr().out


def test_cli_rejects_empty_text(capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(""))
    assert main(["analyze"]) == 2
