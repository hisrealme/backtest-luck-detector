"""The example notebook, checked without a kernel and without a network.

`examples/01_quickstart.ipynb` is the one artefact in this repository that CI
cannot execute. Executing it needs real SPY prices, and design principle 2 is
that no test touches the network — so the notebook would otherwise be the single
file able to rot unnoticed between releases, which is exactly the failure the
manim layer was cut in `BLUEPRINT.md` §6a to avoid.

Three properties are checkable from the JSON alone, and together they cover the
ways it can actually go wrong.

**Its imports still resolve.** An API rename is the likeliest rot by a wide
margin, and importing every name the notebook imports catches it in
milliseconds, offline.

**Its stored outputs are the published numbers.** The notebook ships executed —
see the data note in its first markdown cell — so that a reader without a network
still sees the SPY result, and sees it produced by this code path rather than
transcribed into it. That only holds while the stored outputs *are* the published
numbers, so they are pinned here. If someone re-runs the notebook against a
different window and commits the result, this fails rather than quietly shipping
a README and a notebook that disagree.

**It carries no market data.** The stored outputs are printed statistics, and
that is the whole of what may be committed. No price series, no CSV, in keeping
with the rule that no market data enters this repository.
"""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

NOTEBOOK = Path(__file__).resolve().parents[2] / "examples" / "01_quickstart.ipynb"

#: Every figure the README and `docs/METHODS.md` §10 quote for the SPY run. The
#: notebook recomputes them; this asserts it still lands on the same ones.
PUBLISHED: tuple[str, ...] = (
    "MA(80,250)",
    "0.491",  # winner's annualised Sharpe
    "0.9764",  # naive PSR — the test that passes
    "0.309",  # expected maximum Sharpe of noise
    "0.7692",  # Deflated Sharpe Ratio
    "0.715",  # the Sharpe the winner actually needed
    "0.8396",  # PBO
    "0.2428",  # Reality Check against zero
    "0 of 157",  # variants beating buy-and-hold
    "LIKELY_LUCK",
    "LIKELY_SKILL",  # the control, without which this is a rubber stamp
)

_IMPORT = re.compile(r"^\s*from\s+(luckdetector[\w.]*)\s+import\s+(.+?)\s*$", re.MULTILINE)


@pytest.fixture(scope="module")
def notebook() -> dict[str, Any]:
    assert NOTEBOOK.is_file(), f"{NOTEBOOK} is missing"
    parsed: dict[str, Any] = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return parsed


def code_cells(notebook: dict[str, Any]) -> list[str]:
    return [
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]


def stored_output(notebook: dict[str, Any]) -> str:
    """Every text output stored in the file, concatenated."""
    chunks: list[str] = []
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            chunks.extend(output.get("text", []))
            plain = output.get("data", {}).get("text/plain", [])
            chunks.extend(plain)
    return "".join(chunks)


class TestItStillRuns:
    def test_every_name_it_imports_from_the_package_exists(
        self, notebook: dict[str, Any]
    ) -> None:
        """The cheap half of executing it: resolve the API surface it touches."""
        found = 0
        for source in code_cells(notebook):
            for module_name, names in _IMPORT.findall(source):
                module = importlib.import_module(module_name)
                for name in (part.strip() for part in names.split(",")):
                    assert hasattr(module, name), f"{module_name} has no {name!r}"
                    found += 1
        assert found > 10, "notebook no longer imports from luckdetector — has it been gutted?"

    def test_no_cell_stored_an_error(self, notebook: dict[str, Any]) -> None:
        for cell in notebook["cells"]:
            for output in cell.get("outputs", []):
                assert output["output_type"] != "error", output.get("evalue")

    def test_every_code_cell_was_executed(self, notebook: dict[str, Any]) -> None:
        """A half-run notebook is worse than an unrun one: it looks complete."""
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code" and "".join(cell["source"]).strip():
                assert cell["execution_count"] is not None


class TestItAgreesWithWhatIsPublished:
    @pytest.mark.parametrize("figure", PUBLISHED)
    def test_the_stored_outputs_carry_the_published_figure(
        self, notebook: dict[str, Any], figure: str
    ) -> None:
        assert figure in stored_output(notebook)

    def test_it_resolves_prices_the_way_the_demo_does(self, notebook: dict[str, Any]) -> None:
        """Cache, then download, then refuse — not a private path of its own.

        The alternative considered was reading ``outputs/`` directly, which is
        gitignored and therefore empty in a fresh clone: the notebook would have
        failed for every reader who had not already run the CLI.
        """
        source = "\n".join(code_cells(notebook))
        assert "resolve_demo_prices" in source
        assert "synthetic_prices" not in source, "a quickstart must not headline generated prices"

    def test_it_reached_the_verdict_on_real_prices(self, notebook: dict[str, Any]) -> None:
        """``(cache)`` or ``(download)`` in the output, never a synthetic path."""
        assert re.search(r"SPY\s+\d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}", stored_output(notebook))
        assert "SYNTHETIC" not in stored_output(notebook)


class TestItShipsNoMarketData:
    def test_the_notebook_holds_no_price_series(self, notebook: dict[str, Any]) -> None:
        """Statistics may be stored. Prices may not, in any file, ever."""
        text = stored_output(notebook)
        # A committed price series would show as a long run of comma- or
        # newline-separated numbers; printed statistics never do.
        assert not re.search(r"(\d+\.\d+[,\s]+){50}", text)
        assert ".csv" not in "\n".join(code_cells(notebook))

    def test_it_leaks_no_absolute_paths(self) -> None:
        text = NOTEBOOK.read_text(encoding="utf-8")
        assert "/Users/" not in text
        assert "/home/" not in text
