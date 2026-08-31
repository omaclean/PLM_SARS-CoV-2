"""Every module that reaches ``constants``/``common`` by a fallback chain must land
on the SAME module, whichever arm of the chain fires.

Why this file exists
--------------------
``run_escott.py`` and ``jet_surrogate.py`` each open with a three-way import of
``prescott_iav.constants``:

    1. ``from . import constants``                  -- package member (production)
    2. ``from prescott_iav import constants``       -- ``scripts/`` on sys.path
    3. ``spec_from_file_location`` on the sibling   -- last resort, off disk

``run_escott.py`` additionally reaches ``common`` three ways, and *overwrites*
``LINEAGE_TAGS`` from it when it lands.  Only one arm can execute per interpreter,
so a normal test run leaves arms 2 and 3 unexecuted -- they were the single largest
honest coverage gap in this suite after the two adversarial passes.

The risk is not a crash.  A crash would be loud.  The risk is an arm resolving to a
*different* ``constants`` -- a stale copy elsewhere on ``sys.path``, or a shadowing
top-level ``constants.py`` -- because stage A writes frequency files whose names come
from ``constants`` and stage C has to find those exact names.  A desynchronised
parent map would silently score the wrong ancestor with no error anywhere.  So each
arm is forced to fire here and its resolved values compared against the literals.

``test_common.py::TestConstantsImportFallbacks`` does exactly this for ``common.py``;
this file is the same idea for the two modules it was never done for.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
PRESCOTT_IAV_DIR = SCRIPTS_DIR / "prescott_iav"
for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from prescott_iav import constants  # noqa: E402
from prescott_iav import jet_surrogate as jet_pkg  # noqa: E402
from prescott_iav import run_escott as escott_pkg  # noqa: E402

# The literal the whole pipeline agrees on.  Written out rather than imported so a
# corrupted constants.py cannot make this file agree with itself.
EXPECTED_PARENT_MAP = {
    "J_int": "G.1",
    "J.2_int": "J_int",
    "J.2.4": "J.2_int",
    "K": "J.2.4",
}
EXPECTED_LINEAGE_TAGS = {
    "G.1": "G1",
    "J_int": "J",
    "J.2_int": "J2",
    "J.2.4": "J24",
    "K": "K",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _load_bare(name: str, path: Path):
    """Execute ``path`` as a top-level module.

    ``spec_from_file_location`` with a dotless name leaves ``__package__`` empty, so
    ``from . import constants`` raises ``ImportError`` and arm 1 of the chain is
    skipped -- which is precisely how these files behave when run as scripts.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _block_package_import(monkeypatch):
    """Make arm 2 (``from prescott_iav import constants``) fail too.

    ``sys.modules[name] = None`` is the documented way to force ``import name`` to
    raise ``ImportError`` without touching the filesystem.
    """
    monkeypatch.setitem(sys.modules, "prescott_iav", None)
    monkeypatch.setitem(sys.modules, "prescott_iav.constants", None)


def _drop_prescott_iav_from_syspath(monkeypatch):
    """Remove the directory that would let a bare ``import constants`` succeed."""
    kept = [e for e in sys.path if Path(e or ".").name != "prescott_iav"]
    monkeypatch.setattr(sys, "path", kept)


# --------------------------------------------------------------------------- #
# run_escott.py
# --------------------------------------------------------------------------- #

@pytest.mark.unit
class TestRunEscottConstantsFallbacks:
    """All three arms of ``run_escott``'s constants chain resolve identically."""

    PATH = PRESCOTT_IAV_DIR / "run_escott.py"

    def test_arm1_package_import_is_what_production_uses(self):
        """The imported-as-a-package module is the reference every arm is judged by."""
        assert escott_pkg._constants is constants
        assert escott_pkg.LINEAGE_TAGS == EXPECTED_LINEAGE_TAGS

    def test_arm2_resolves_via_the_prescott_iav_package_on_syspath(self, monkeypatch):
        """No ``__package__`` -> arm 1 fails; ``scripts/`` on sys.path -> arm 2 wins."""
        monkeypatch.syspath_prepend(str(SCRIPTS_DIR))
        bare = _load_bare("run_escott_bare_arm2", self.PATH)
        assert bare._constants.DEFAULT_PARENT_MAPS["clade_evidence"] == EXPECTED_PARENT_MAP
        assert bare.LINEAGE_TAGS == EXPECTED_LINEAGE_TAGS

    def test_arm3_last_resort_loads_constants_straight_off_disk(self, monkeypatch):
        """Arms 1 and 2 blocked -> ``spec_from_file_location`` on the sibling file."""
        _block_package_import(monkeypatch)
        bare = _load_bare("run_escott_bare_arm3", self.PATH)
        assert bare._constants.DEFAULT_PARENT_MAPS["clade_evidence"] == EXPECTED_PARENT_MAP
        # The corrected ladder specifically: K descends from J.2.4, never J.2_int.
        assert bare._constants.DEFAULT_PARENT_MAPS["clade_evidence"]["K"] == "J.2.4"

    @pytest.mark.parametrize("blocked", [False, True])
    def test_every_arm_exports_the_same_naming_helpers(self, monkeypatch, blocked):
        """The names stage A and stage C must agree on byte-for-byte.

        Stage A writes ``K_parentJ24_frequency.txt``; stage C has to find that exact
        file.  If an arm resolved to a different ``constants`` these helpers would
        diverge and the frequency prior would silently go missing.
        """
        if blocked:
            _block_package_import(monkeypatch)
        else:
            monkeypatch.syspath_prepend(str(SCRIPTS_DIR))
        bare = _load_bare(f"run_escott_bare_helpers_{int(blocked)}", self.PATH)
        labels = ("K", "J.2.4", "J.2_int", "J_int", "G.1")
        for label in labels:
            assert bare.variant_parent_token(label) == \
                escott_pkg.variant_parent_token(label), label
            for parent in labels:
                assert bare.alternate_frequency_basename(label, parent) == \
                    escott_pkg.alternate_frequency_basename(label, parent), (label, parent)
        spec = "K=J.2_int,J.2.4=J_int"
        assert bare.parse_edge_spec(spec) == escott_pkg.parse_edge_spec(spec)

    def test_common_fallback_still_supplies_lineage_tags(self, monkeypatch):
        """``run_escott`` overwrites its local ``LINEAGE_TAGS`` from ``common``.

        The local literal and ``common``'s must stay equal; this pins that they do,
        via the bare-module arm where ``from . import common`` cannot fire.
        """
        monkeypatch.syspath_prepend(str(PRESCOTT_IAV_DIR))
        bare = _load_bare("run_escott_bare_common", self.PATH)
        assert bare._common is not None, "the bare arm should still find common.py"
        assert bare.LINEAGE_TAGS == EXPECTED_LINEAGE_TAGS
        assert bare.LINEAGE_TAGS == dict(bare._common.LINEAGE_TAGS)

    def test_a_module_scope_constant_is_identical_across_arms(self, monkeypatch):
        """Spot-check the numeric constants the two hunters added, too."""
        _block_package_import(monkeypatch)
        bare = _load_bare("run_escott_bare_numbers", self.PATH)
        for name in ("NO_FREQUENCY_SENTINEL", "MIN_FREQUENCY_MATCH_FRACTION"):
            assert getattr(bare, name) == getattr(escott_pkg, name), name


# --------------------------------------------------------------------------- #
# jet_surrogate.py
# --------------------------------------------------------------------------- #

@pytest.mark.unit
class TestJetSurrogateConstantsFallbacks:
    """All three arms of ``jet_surrogate``'s constants chain resolve identically.

    This one carries the tuned ``--trace-top-fraction``.  The comment above the chain
    records why: the driver once hard-coded a *different* top fraction and passed it
    down unconditionally, turning a tuned 0.90 into 0.30 and leaving 52.8% of sites
    at ``trace == 0`` -- i.e. pure noise, because pred.R:487 multiplies every ESCOTT
    column by ``trace[i]``.  An arm resolving to a stale constants would do the same
    thing silently, so the value is checked on every arm.
    """

    PATH = PRESCOTT_IAV_DIR / "jet_surrogate.py"

    NAMES = (
        "DEFAULT_TRACE_TOP_FRACTION",
        "MAX_ZERO_TRACE_FRACTION",
        "WARN_ZERO_TRACE_FRACTION",
    )

    def test_arm1_package_import_is_what_production_uses(self):
        assert jet_pkg._constants is constants
        assert jet_pkg.DEFAULT_TRACE_TOP_FRACTION == constants.DEFAULT_TRACE_TOP_FRACTION

    def test_arm2_resolves_via_the_prescott_iav_package_on_syspath(self, monkeypatch):
        monkeypatch.syspath_prepend(str(SCRIPTS_DIR))
        bare = _load_bare("jet_surrogate_bare_arm2", self.PATH)
        for name in self.NAMES:
            assert getattr(bare, name) == getattr(jet_pkg, name), name

    def test_arm3_last_resort_loads_constants_straight_off_disk(self, monkeypatch):
        _block_package_import(monkeypatch)
        _drop_prescott_iav_from_syspath(monkeypatch)
        bare = _load_bare("jet_surrogate_bare_arm3", self.PATH)
        for name in self.NAMES:
            assert getattr(bare, name) == getattr(jet_pkg, name), name

    def test_the_tuned_trace_top_fraction_survives_every_arm(self, monkeypatch):
        """0.90 is a measured value; no arm may quietly serve a different one."""
        _block_package_import(monkeypatch)
        bare = _load_bare("jet_surrogate_bare_tuned", self.PATH)
        assert bare.DEFAULT_TRACE_TOP_FRACTION == pytest.approx(0.9)
        assert 0.0 < bare.MAX_ZERO_TRACE_FRACTION <= 1.0
        assert 0.0 < bare.WARN_ZERO_TRACE_FRACTION <= bare.MAX_ZERO_TRACE_FRACTION


# --------------------------------------------------------------------------- #
# cross-module agreement
# --------------------------------------------------------------------------- #

@pytest.mark.unit
class TestAllModulesSeeOneConstants:

    def test_no_two_modules_hold_different_constants_objects(self):
        """In production every module must share ONE ``constants`` instance."""
        from prescott_iav import common as common_mod

        seen = {
            "common": getattr(common_mod, "_constants", constants),
            "run_escott": escott_pkg._constants,
            "jet_surrogate": jet_pkg._constants,
        }
        for name, mod in seen.items():
            assert mod is constants, f"{name} resolved a different constants module"

    def test_the_corrected_ladder_is_the_default_everywhere(self):
        """K <- J.2.4.  The old K <- J.2_int edge may exist only as a labelled preset."""
        assert constants.DEFAULT_PARENT_MAPS[constants.DEFAULT_PARENT_MAP_PRESET] == \
            EXPECTED_PARENT_MAP
        alt = [k for k, v in constants.DEFAULT_PARENT_MAPS.items()
               if v.get("K") == "J.2_int"]
        assert constants.DEFAULT_PARENT_MAP_PRESET not in alt, \
            "the superseded K <- J.2_int edge must never be the default"
