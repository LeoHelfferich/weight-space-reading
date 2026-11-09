"""Repo hygiene: no secrets, no pickled arrays, probe outputs stay untracked."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY_RE = re.compile(r"sk-[A-Za-z0-9_\-]{12,}")


def _source_files():
    for folder in ("src", "configs"):
        for path in (ROOT / folder).rglob("*"):
            if path.suffix in {".py", ".json", ".md"}:
                yield path


def test_no_secret_material_in_the_tree():
    for path in _source_files():
        text = path.read_text()
        assert KEY_RE.search(text) is None, path
        assert "BEGIN PRIVATE KEY" not in text
        assert "import pickle" not in text
        assert "np.load(" not in text or "allow_pickle=False" in text


def test_probe_outputs_are_gitignored():
    ignore = (ROOT / ".gitignore").read_text()
    assert "results/probe_cache/" in ignore
    assert "results/probe_runs/" in ignore
    assert ".env" in ignore
