#!/usr/bin/env python3
"""Tests for ``scripts/prescott_iav/common.py`` and ``scripts/prescott_iav/constants.py``.

``common.py`` is the module every other stage-1 module imports, and almost
nothing in it fails loudly.  A wrong lineage key produces a file with a slightly
different name (and a "cache miss" that silently recomputes from the wrong
panel); a wrong consensus column map shifts the frequency prior against the
scores by one residue; a preset that regressed to the old ``K <- J.2_int`` edge
builds the prior from a lineage that is not the parent at all.  None of those
raise.  So the tests below are organised around *what a silent drift would look
like* rather than around the public API:

* **naming** -- ``safe_label`` / ``dot_free_key`` / ``lineage_tag`` /
  ``escott_prot_token`` are four different flavours of "key", each with its own
  legal alphabet, and the whole tree is named by them.  Dots survive one and are
  destroyed by another, on purpose (``prescott.py:902`` runs ``os.path.splitext``
  on its ``-o`` value), and the transforms are lossy, so their *injectivity over
  the real label set* is the property that matters and is asserted here;
* **cache keys** -- ``md5_text`` / ``md5_file`` / ``write_json`` are what decide
  whether an expensive stage is rerun.  ``write_json`` must be byte-deterministic
  across dict insertion orders or every manifest comparison is a coin flip;
* **the parent map** -- every error branch of ``resolve_parent_map``, including
  cycle detection, because a bad edge does not fail later, it just quietly uses
  the wrong panel;
* **mirror fidelity** -- ``safe_label``, ``translate_reference_cds`` and
  ``build_consensus_and_column_map`` claim to be byte-faithful mirrors of
  functions in ``Functions_HuggingFace``.  ``TestMirrorFidelityAgainstRMA``
  imports the originals and compares, including a seeded randomised differential
  over 120 alignments.  Those tests carry ``requires_rma`` (the originals need
  torch); everything else in this file is pure, offline and sub-second.

Expected values are literals, hand arithmetic, or computed by an independent
code path in ``conftest`` -- never by the function under test.

Run with::

    /home3/oml4h/miniconda3/envs/PRESCOTT/bin/python -m pytest \
        /home3/oml4h/PLM_SARS-CoV-2/tests_prescott_iav/test_common.py -q
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _p in (str(SCRIPTS_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from prescott_iav import common, constants  # noqa: E402

from tests_prescott_iav.conftest import (  # noqa: E402
    AA20,
    CONTESTED_EDGE,
    EXPECTED_INPUT_ONLY_LINEAGES,
    EXPECTED_LINEAGE_TAGS,
    EXPECTED_MAX_ZERO_TRACE_FRACTION,
    EXPECTED_PARENT_MAP,
    EXPECTED_PROT_TOKENS,
    EXPECTED_SENSITIVITY_PARENT_MAP,
    EXPECTED_TRACE_TOP_FRACTION,
    EXPECTED_WARN_ZERO_TRACE_FRACTION,
    LINEAGE_ORDER,
    PLM_CACHE_ROW_ORDER,
    QUERY_CDS,
    QUERY_HEADER,
    QUERY_PROTEIN,
)
from tests_prescott_iav.conftest import md5_text as ref_md5_text  # noqa: E402
from tests_prescott_iav.conftest import read_fasta as ref_read_fasta  # noqa: E402
from tests_prescott_iav.conftest import translate_cds as ref_translate_cds  # noqa: E402
from tests_prescott_iav.conftest import write_fasta as ref_write_fasta  # noqa: E402

pytestmark = pytest.mark.unit


# The three ``ignore`` characters, written out rather than imported, so a test
# comparing against them is testing the module and not comparing it with itself.
EXPECTED_IGNORE_CHARS = frozenset({"-", "*", "."})


# =========================================================================== #
# constants.py -- the single source of truth
# =========================================================================== #

class TestConstantsModule:
    """``constants.py`` is THE authority; every literal here is independent of it."""

    def test_default_preset_is_the_corrected_ladder(self):
        assert constants.DEFAULT_PARENT_MAP_PRESET == "clade_evidence"
        assert constants.DEFAULT_PARENT_MAPS["clade_evidence"] == EXPECTED_PARENT_MAP

    def test_k_descends_from_j24_not_j2_int(self):
        """The single contested edge, named explicitly so a regression is unmissable."""
        child, default_parent, sensitivity_parent = CONTESTED_EDGE
        assert constants.DEFAULT_PARENT_MAPS["clade_evidence"][child] == default_parent
        assert constants.DEFAULT_PARENT_MAPS["clade_evidence"][child] != sensitivity_parent

    def test_brief_as_stated_is_retained_only_as_the_alternative(self):
        assert constants.DEFAULT_PARENT_MAPS["brief_as_stated"] == EXPECTED_SENSITIVITY_PARENT_MAP
        assert constants.DEFAULT_PARENT_MAP_PRESET != "brief_as_stated"

    def test_preset_names_are_exactly_the_two_known_presets(self):
        assert constants.preset_names() == ("clade_evidence", "brief_as_stated")

    def test_the_ladder_is_linear_and_rooted_at_g1(self):
        """G.1 -> J_int -> J.2_int -> J.2.4 -> K: every node has exactly one child."""
        ladder = constants.DEFAULT_PARENT_MAPS["clade_evidence"]
        parents = list(ladder.values())
        assert len(parents) == len(set(parents)), "a linear ladder cannot reuse a parent"
        roots = [p for p in parents if p not in ladder]
        assert roots == ["G.1"]
        assert set(ladder) | set(parents) == set(LINEAGE_ORDER)

    def test_input_only_lineages(self):
        assert constants.INPUT_ONLY_LINEAGES == EXPECTED_INPUT_ONLY_LINEAGES
        assert isinstance(constants.INPUT_ONLY_LINEAGES, frozenset)

    def test_lineage_tags(self):
        assert constants.LINEAGE_TAGS == EXPECTED_LINEAGE_TAGS

    def test_trace_defaults(self):
        assert constants.DEFAULT_TRACE_TOP_FRACTION == EXPECTED_TRACE_TOP_FRACTION
        assert constants.MAX_ZERO_TRACE_FRACTION == EXPECTED_MAX_ZERO_TRACE_FRACTION
        assert constants.WARN_ZERO_TRACE_FRACTION == EXPECTED_WARN_ZERO_TRACE_FRACTION
        # The warn band must be a real band inside the refusal ceiling.
        assert 0.0 < constants.WARN_ZERO_TRACE_FRACTION < constants.MAX_ZERO_TRACE_FRACTION < 1.0

    def test_blat_reference_paths_are_absolute(self):
        for value in (constants.BLAT_REFERENCE_JET_RES,
                      constants.BLAT_REFERENCE_MSA,
                      constants.BLAT_REFERENCE_PDB):
            assert Path(value).is_absolute()

    def test_all_exports_exist(self):
        missing = [name for name in constants.__all__ if not hasattr(constants, name)]
        assert missing == []

    def test_constants_imports_no_sibling(self):
        """"Nothing in this file may import a sibling module" -- enforced, not hoped."""
        source = (SCRIPTS_DIR / "prescott_iav" / "constants.py").read_text(encoding="utf-8")
        for sibling in ("common", "jet_surrogate", "prepare_inputs",
                        "run_escott", "leakage_check"):
            assert f"import {sibling}" not in source
            assert f"from .{sibling}" not in source
        # ... and it must not need anything outside the stdlib.
        assert "import numpy" not in source and "import pandas" not in source


class TestCommonReexportsAreBindings:
    """``common`` must *bind* the constants, not copy them: one dict, one truth."""

    @pytest.mark.parametrize(
        "common_name, constants_name",
        [
            ("PARENT_MAP_PRESETS", "DEFAULT_PARENT_MAPS"),
            ("DEFAULT_PARENT_MAP_PRESET", "DEFAULT_PARENT_MAP_PRESET"),
            ("INPUT_ONLY_LINEAGES", "INPUT_ONLY_LINEAGES"),
            ("LINEAGE_TAGS", "LINEAGE_TAGS"),
            ("variant_parent_token", "variant_parent_token"),
            ("alternate_frequency_basename", "alternate_frequency_basename"),
            ("sensitivity_edges_between_presets", "sensitivity_edges_between_presets"),
            ("parse_edge_spec", "parse_edge_spec"),
            ("DEFAULT_TRACE_TOP_FRACTION", "DEFAULT_TRACE_TOP_FRACTION"),
            ("MAX_ZERO_TRACE_FRACTION", "MAX_ZERO_TRACE_FRACTION"),
            ("WARN_ZERO_TRACE_FRACTION", "WARN_ZERO_TRACE_FRACTION"),
        ],
    )
    def test_is_the_same_object(self, common_name, constants_name):
        assert getattr(common, common_name) is getattr(constants, constants_name)

    def test_common_has_no_second_copy_of_the_parent_map(self):
        """A literal ladder re-typed in common.py is exactly the drift constants.py exists to stop."""
        source = (SCRIPTS_DIR / "prescott_iav" / "common.py").read_text(encoding="utf-8")
        code = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        assert '"J.2.4":' not in code and "'J.2.4':" not in code

    def test_alphabets(self):
        assert common.STANDARD_AMINO_ACIDS == tuple(AA20)
        assert common.GEMME_AA_ORDER == tuple(AA20)
        assert common.PLM_AA_ORDER == PLM_CACHE_ROW_ORDER
        assert common.IGNORE_ALIGNMENT_CHARS == EXPECTED_IGNORE_CHARS
        # GEMME and PLM hold the same 20 letters in DIFFERENT orders; a test that
        # only checked set equality would not notice them being swapped.
        assert set(common.GEMME_AA_ORDER) == set(common.PLM_AA_ORDER)
        assert common.GEMME_AA_ORDER != common.PLM_AA_ORDER


class TestConstantsImportFallbacks:
    """``common.py`` reaches ``constants`` three ways; all three must land on it.

    The package-relative import is what the pipeline uses.  The other two exist
    because ``prescott_iav/`` is also put on ``sys.path`` directly (stage-1
    modules are runnable as scripts), and because a copied tree may have neither.
    If a fallback silently produced a *different* constants module the two halves
    of the pipeline would disagree about the parent map with no error anywhere --
    so each path is loaded here and its ladder checked against the literal.
    """

    COMMON_PATH = SCRIPTS_DIR / "prescott_iav" / "common.py"

    @staticmethod
    def _load_bare(name):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            name, TestConstantsImportFallbacks.COMMON_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)          # __package__ == '' -> no relative import
        return module

    def test_bare_module_falls_back_to_a_top_level_constants(self, monkeypatch):
        """``from . import constants`` fails; ``import constants`` must succeed."""
        monkeypatch.syspath_prepend(str(SCRIPTS_DIR / "prescott_iav"))
        monkeypatch.delitem(sys.modules, "constants", raising=False)
        bare = self._load_bare("prescott_iav_common_bare_a")
        assert bare.PARENT_MAP_PRESETS["clade_evidence"] == EXPECTED_PARENT_MAP
        assert bare.DEFAULT_TRACE_TOP_FRACTION == EXPECTED_TRACE_TOP_FRACTION

    def test_last_resort_loads_constants_straight_off_disk(self, monkeypatch):
        """Both import forms blocked -> ``spec_from_file_location`` on the sibling."""
        # sys.modules[name] = None makes `import name` raise ImportError.
        monkeypatch.setitem(sys.modules, "constants", None)
        for entry in list(sys.path):
            if entry.endswith("prescott_iav"):
                monkeypatch.delitem(sys.path, sys.path.index(entry), raising=False)
        bare = self._load_bare("prescott_iav_common_bare_b")
        assert bare.PARENT_MAP_PRESETS["clade_evidence"] == EXPECTED_PARENT_MAP
        assert bare.PARENT_MAP_PRESETS["clade_evidence"]["K"] == "J.2.4"
        assert bare.LINEAGE_TAGS == EXPECTED_LINEAGE_TAGS

    def test_every_path_yields_the_same_values(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "constants", None)
        bare = self._load_bare("prescott_iav_common_bare_c")
        for name in ("PARENT_MAP_PRESETS", "LINEAGE_TAGS", "INPUT_ONLY_LINEAGES",
                     "DEFAULT_PARENT_MAP_PRESET", "DEFAULT_TRACE_TOP_FRACTION",
                     "MAX_ZERO_TRACE_FRACTION", "WARN_ZERO_TRACE_FRACTION"):
            assert getattr(bare, name) == getattr(common, name), name


# =========================================================================== #
# safe_label
# =========================================================================== #

class TestSafeLabel:

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("K", "K"),
            ("J.2_int", "J.2_int"),        # dots SURVIVE -- that is the point
            ("J.2.4", "J.2.4"),
            ("  K  ", "K"),                 # outer whitespace stripped
            ("\tK\n", "K"),
            ("A/H3N2", "A-H3N2"),           # '/' would open a directory
            ("a b c", "a_b_c"),
            ("A/B C/D", "A-B_C-D"),
            ("", ""),
            ("   ", ""),
            ("2024-25", "2024-25"),
            ("A//B", "A--B"),
            ("A  B", "A__B"),               # each space individually
        ],
    )
    def test_transform(self, raw, expected):
        assert common.safe_label(raw) == expected

    def test_is_idempotent_for_every_real_label(self):
        for label in LINEAGE_ORDER:
            once = common.safe_label(label)
            assert common.safe_label(once) == once

    def test_real_labels_are_unchanged(self):
        """No production label contains a space or a slash, so the key IS the label."""
        assert {label: common.safe_label(label) for label in LINEAGE_ORDER} == {
            label: label for label in LINEAGE_ORDER
        }

    def test_is_injective_over_the_real_label_set(self):
        keys = [common.safe_label(label) for label in LINEAGE_ORDER]
        assert len(set(keys)) == len(LINEAGE_ORDER)

    def test_is_not_injective_in_general(self):
        """Documented lossiness: ' ' and '/' collapse onto '_' and '-'."""
        assert common.safe_label("A B") == common.safe_label("A_B") == "A_B"
        assert common.safe_label("A/B") == common.safe_label("A-B") == "A-B"

    def test_inner_whitespace_other_than_space_is_not_normalised(self):
        """A tab INSIDE the label survives into the filename.  Pinned, not endorsed."""
        assert common.safe_label("A\tB") == "A\tB"

    def test_non_string_raises(self):
        with pytest.raises(AttributeError):
            common.safe_label(None)  # type: ignore[arg-type]


# =========================================================================== #
# dot_free_key and the splitext trap it exists to defuse
# =========================================================================== #

class TestDotFreeKey:

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("G.1", "G_1"),
            ("J_int", "J_int"),
            ("J.2_int", "J_2_int"),
            ("J.2.4", "J_2_4"),
            ("K", "K"),
            ("", ""),
            (".", "_"),
            ("..", "__"),
        ],
    )
    def test_transform(self, raw, expected):
        assert common.dot_free_key(raw) == expected

    def test_common_and_constants_agree(self):
        for label in LINEAGE_ORDER:
            assert common.dot_free_key(label) == constants.dot_free_key(label)

    def test_output_never_contains_a_dot(self):
        for label in LINEAGE_ORDER:
            assert "." not in common.dot_free_key(label)

    def test_splitext_truncates_a_dotted_key_and_not_a_dot_free_one(self):
        """The whole reason ``dot_free_key`` exists (prescott.py:902 splitext its -o)."""
        assert os.path.splitext("J.2_int") == ("J", ".2_int")   # would become 'J'
        assert os.path.splitext("J.2.4") == ("J.2", ".4")        # would become 'J.2'
        for label in LINEAGE_ORDER:
            key = common.dot_free_key(label)
            assert os.path.splitext(key) == (key, "")

    def test_is_injective_over_the_real_label_set(self):
        keys = [common.dot_free_key(label) for label in LINEAGE_ORDER]
        assert sorted(keys) == sorted(["G_1", "J_int", "J_2_int", "J_2_4", "K"])
        assert len(set(keys)) == len(LINEAGE_ORDER)

    def test_is_not_injective_in_general_and_is_not_invertible(self):
        """``J.2_int`` and a hypothetical ``J_2_int`` collapse onto one key.

        There is therefore NO way to recover the label from a filename; the
        manifest's ``frequency_index`` carries the label explicitly for exactly
        this reason.
        """
        assert common.dot_free_key("J.2_int") == common.dot_free_key("J_2_int") == "J_2_int"
        assert common.dot_free_key("J.2.4") == common.dot_free_key("J_2_4") == "J_2_4"

    def test_dotted_lineage_keys_round_trip_through_a_filename(self):
        """``safe_label`` keys keep their dots, so the SUFFIX must be stripped by name.

        ``Path.stem`` is safe here only because every suffix we use is a real
        extension; ``Path('J.2.4_query.fasta').stem`` keeps both dots.
        """
        for label in LINEAGE_ORDER:
            key = common.safe_label(label)
            name = Path(f"{key}_query.fasta")
            assert name.suffix == ".fasta"
            assert name.stem == f"{key}_query"
            assert name.stem[: -len("_query")] == key == label


# =========================================================================== #
# lineage_tag
# =========================================================================== #

class TestLineageTag:

    @pytest.mark.parametrize("label, tag", sorted(EXPECTED_LINEAGE_TAGS.items()))
    def test_pinned_tags(self, label, tag):
        assert common.lineage_tag(label) == tag

    def test_pinned_tags_beat_the_regex_fallback(self):
        """``J_int`` -> ``J`` only because it is pinned; the fallback would say ``Jint``."""
        assert common.lineage_tag("J_int") == "J"
        assert common.lineage_tag("J_intX") == "JintX"

    @pytest.mark.parametrize(
        "label, tag",
        [
            ("L.1.2", "L12"),
            ("A/H3N2", "AH3N2"),
            ("2024-25", "202425"),
            ("x", "x"),
            ("a_b_c", "abc"),
            ("K ", "K"),          # trailing space is stripped BY THE REGEX, not by strip()
        ],
    )
    def test_regex_fallback(self, label, tag):
        assert label not in common.LINEAGE_TAGS
        assert common.lineage_tag(label) == tag

    @pytest.mark.parametrize("label", ["", "...", "___", "  ", "-/-", "!!"])
    def test_untaggable_labels_raise(self, label):
        with pytest.raises(ValueError, match="Cannot derive an escott tag"):
            common.lineage_tag(label)

    def test_error_message_names_the_offending_label(self):
        with pytest.raises(ValueError) as excinfo:
            common.lineage_tag("...")
        assert "'...'" in str(excinfo.value)

    def test_tags_are_injective_over_the_real_label_set(self):
        tags = [common.lineage_tag(label) for label in LINEAGE_ORDER]
        assert len(set(tags)) == len(LINEAGE_ORDER)

    def test_tag_collision_is_possible_off_the_pinned_set(self):
        """``J.2.4`` and a hypothetical ``J24`` produce the SAME tag.

        Harmless today because ``J24`` is not a lineage, but it is why the tags
        are pinned in ``constants.LINEAGE_TAGS`` rather than derived.
        """
        assert common.lineage_tag("J.2.4") == common.lineage_tag("J24") == "J24"

    def test_every_tag_yields_the_expected_escott_prot_token(self):
        for label in LINEAGE_ORDER:
            header = f"HA{common.lineage_tag(label)}"
            assert common.escott_prot_token(header) == header
            assert header == EXPECTED_PROT_TOKENS[label]


# =========================================================================== #
# escott_prot_token
# =========================================================================== #

class TestEscottProtToken:

    @pytest.mark.parametrize(
        "header, token",
        [
            ("HAK", "HAK"),
            ("HAJ2", "HAJ2"),
            ("EPI4748783|HA|A/England/01837755/2025|EPI_ISL_20210731|J.2.4.1", "EPI4748783"),
            ("HA_J", "HA"),          # '_' is NOT alphanumeric to escott
            ("HA J", "HA"),
            ("HA.J", "HA"),
            ("HA-J", "HA"),
            ("abc123XYZ", "abc123XYZ"),
        ],
    )
    def test_split_at_first_non_alphanumeric(self, header, token):
        assert common.escott_prot_token(header) == token

    @pytest.mark.parametrize("header", ["|abc", " abc", "_abc", ".abc", ""])
    def test_leading_non_alphanumeric_yields_an_empty_token(self, header):
        """escott would then name every output ``_normPred_evolCombi.txt``.

        ``prepare_inputs`` guards against this by asserting ``token == header``
        before writing the query, which is the guard this test underwrites.
        """
        assert common.escott_prot_token(header) == ""

    def test_underscore_is_why_the_j_int_tag_is_not_jint(self):
        """If the header were ``HAJ_int``, escott would silently call it ``HAJ``."""
        assert common.escott_prot_token("HAJ_int") == "HAJ"
        assert common.escott_prot_token("HAJ") == "HAJ"

    def test_token_is_a_prefix_of_the_header(self):
        for header in ("HAK", "EPI1|HA", "abc-def", "x"):
            assert header.startswith(common.escott_prot_token(header))


# =========================================================================== #
# Hashing and cache keys
# =========================================================================== #

class TestHashing:

    def test_md5_text_matches_hashlib(self):
        assert common.md5_text("") == hashlib.md5(b"").hexdigest()
        assert common.md5_text(QUERY_PROTEIN) == ref_md5_text(QUERY_PROTEIN)
        assert common.md5_text("") == "d41d8cd98f00b204e9800998ecf8427e"

    def test_md5_text_is_utf8_not_latin1(self):
        assert common.md5_text("é") == hashlib.md5("é".encode("utf-8")).hexdigest()
        assert common.md5_text("é") != hashlib.md5("é".encode("latin-1")).hexdigest()

    def test_md5_text_is_case_and_whitespace_sensitive(self):
        assert common.md5_text("ACD") != common.md5_text("acd")
        assert common.md5_text("ACD") != common.md5_text("ACD ")

    def test_md5_file_matches_hashlib(self, tmp_path):
        path = tmp_path / "blob.bin"
        payload = bytes(range(256)) * 37
        path.write_bytes(payload)
        assert common.md5_file(path) == hashlib.md5(payload).hexdigest()

    def test_md5_file_of_an_empty_file(self, tmp_path):
        path = tmp_path / "empty.bin"
        path.write_bytes(b"")
        assert common.md5_file(path) == "d41d8cd98f00b204e9800998ecf8427e"

    @pytest.mark.parametrize("chunk_size", [1, 3, 7, 1 << 20])
    def test_md5_file_is_chunk_size_independent(self, tmp_path, chunk_size):
        path = tmp_path / "chunked.bin"
        payload = b"".join(bytes([i % 251]) for i in range(1000))
        path.write_bytes(payload)
        assert common.md5_file(path, chunk_size=chunk_size) == hashlib.md5(payload).hexdigest()

    def test_md5_file_is_binary_not_text(self, tmp_path):
        """CRLF must NOT be normalised: the byte stream is the cache key."""
        crlf = tmp_path / "crlf.txt"
        lf = tmp_path / "lf.txt"
        crlf.write_bytes(b">a\r\nACD\r\n")
        lf.write_bytes(b">a\nACD\n")
        assert common.md5_file(crlf) != common.md5_file(lf)

    def test_md5_file_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            common.md5_file(tmp_path / "absent")

    def test_md5_text_and_md5_file_agree_on_the_same_bytes(self, tmp_path):
        path = tmp_path / "seq.txt"
        path.write_text(QUERY_PROTEIN, encoding="utf-8")
        assert common.md5_file(path) == common.md5_text(QUERY_PROTEIN)


class TestJsonIO:

    def test_round_trip(self, tmp_path):
        payload = {"b": 2, "a": [1, 2, 3], "c": {"nested": True}}
        path = tmp_path / "m.json"
        common.write_json(path, payload)
        assert common.read_json(path) == payload

    def test_creates_missing_parents(self, tmp_path):
        path = tmp_path / "deep" / "deeper" / "m.json"
        common.write_json(path, {"x": 1})
        assert path.exists()
        assert common.read_json(path) == {"x": 1}

    def test_bytes_are_deterministic_across_insertion_order(self, tmp_path):
        """The manifest IS the cache key; two equal dicts must give equal bytes."""
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        common.write_json(a, {"z": 1, "a": 2, "m": 3})
        common.write_json(b, {"m": 3, "a": 2, "z": 1})
        assert a.read_bytes() == b.read_bytes()
        assert common.md5_file(a) == common.md5_file(b)

    def test_layout_is_sorted_indent_two_with_a_trailing_newline(self, tmp_path):
        path = tmp_path / "m.json"
        common.write_json(path, {"z": 1, "a": 2})
        text = path.read_text(encoding="utf-8")
        assert text == '{\n  "a": 2,\n  "z": 1\n}\n'

    def test_non_serialisable_values_become_strings(self, tmp_path):
        path = tmp_path / "m.json"
        common.write_json(path, {"p": Path("/a/b"), "s": {1, 2}})
        loaded = common.read_json(path)
        assert loaded["p"] == "/a/b"
        assert loaded["s"] in ("{1, 2}", "{2, 1}")

    def test_overwrites_rather_than_appends(self, tmp_path):
        path = tmp_path / "m.json"
        common.write_json(path, {"a": 1, "b": 2})
        common.write_json(path, {"a": 1})
        assert common.read_json(path) == {"a": 1}
        assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}

    def test_read_json_missing_returns_none(self, tmp_path):
        assert common.read_json(tmp_path / "absent.json") is None

    @pytest.mark.parametrize("text", ["", "{not json", '{"a": 1', "[1, 2,]", "\x00\x01"])
    def test_read_json_corrupt_returns_none(self, tmp_path, text):
        """A truncated manifest from a killed job must degrade to 'recompute'."""
        path = tmp_path / "bad.json"
        path.write_text(text, encoding="utf-8")
        assert common.read_json(path) is None

    def test_read_json_on_a_directory_returns_none(self, tmp_path):
        """``exists()`` is True for a directory; the OSError branch must catch it."""
        target = tmp_path / "adir"
        target.mkdir()
        assert common.read_json(target) is None

    def test_read_json_of_a_bare_scalar_is_not_a_dict_but_is_returned(self, tmp_path):
        """Documented looseness: the annotation says Dict, the code returns whatever parsed."""
        path = tmp_path / "scalar.json"
        path.write_text("42", encoding="utf-8")
        assert common.read_json(path) == 42


class TestEnsureDir:

    def test_creates_nested_and_returns_the_path(self, tmp_path):
        target = tmp_path / "a" / "b" / "c"
        result = common.ensure_dir(target)
        assert result == target
        assert target.is_dir()

    def test_is_idempotent(self, tmp_path):
        target = tmp_path / "a"
        common.ensure_dir(target)
        (target / "keep.txt").write_text("x", encoding="utf-8")
        common.ensure_dir(target)
        assert (target / "keep.txt").read_text(encoding="utf-8") == "x"

    def test_raises_when_a_file_occupies_the_path(self, tmp_path):
        target = tmp_path / "occupied"
        target.write_text("x", encoding="utf-8")
        with pytest.raises(FileExistsError):
            common.ensure_dir(target)


# =========================================================================== #
# FASTA I/O
# =========================================================================== #

class TestReadFasta:

    def test_reads_the_conftest_writer_back(self, tmp_path):
        records = [("h1", "ACDEF"), ("h2", "GHIKL")]
        path = ref_write_fasta(tmp_path / "x.fa", records)
        assert list(common.read_fasta(path)) == records

    def test_multiline_sequences_are_joined(self, tmp_path):
        path = ref_write_fasta(tmp_path / "x.fa", [("h", QUERY_PROTEIN)], line_width=7)
        assert list(common.read_fasta(path)) == [("h", QUERY_PROTEIN)]

    def test_sequences_are_uppercased(self, tmp_path):
        path = tmp_path / "x.fa"
        path.write_text(">h\nacdef\nGhIk\n", encoding="utf-8")
        assert list(common.read_fasta(path)) == [("h", "ACDEFGHIK")]

    def test_every_record_is_uppercased_not_just_the_last(self, tmp_path):
        """There are TWO ``yield`` sites -- one mid-loop, one after it.

        A single-record file only ever reaches the second, so it cannot tell the
        two apart; three records exercise both.  ``prepare_inputs`` compares
        panel residues against the uppercase reference, so a lowercase row would
        read as an all-positions difference.
        """
        path = tmp_path / "x.fa"
        path.write_text(">a\nacd\n>b\nefg\n>c\nhik\n", encoding="utf-8")
        assert list(common.read_fasta(path)) == [("a", "ACD"), ("b", "EFG"), ("c", "HIK")]
        assert common.read_fasta_sequences(path) == ["ACD", "EFG", "HIK"]

    def test_header_gt_is_stripped_and_description_kept_whole(self, tmp_path):
        path = tmp_path / "x.fa"
        path.write_text(">  EPI1|HA A/England/1/2025  \nAC\n", encoding="utf-8")
        assert list(common.read_fasta(path)) == [("EPI1|HA A/England/1/2025", "AC")]

    def test_crlf_and_blank_lines(self, tmp_path):
        path = tmp_path / "x.fa"
        path.write_bytes(b">h1\r\n\r\nACD\r\n\r\n>h2\r\nEF\r\n")
        assert list(common.read_fasta(path)) == [("h1", "ACD"), ("h2", "EF")]

    def test_missing_trailing_newline(self, tmp_path):
        path = tmp_path / "x.fa"
        path.write_text(">h\nACD", encoding="utf-8")
        assert list(common.read_fasta(path)) == [("h", "ACD")]

    def test_empty_file_yields_nothing(self, tmp_path):
        path = tmp_path / "x.fa"
        path.write_text("", encoding="utf-8")
        assert list(common.read_fasta(path)) == []

    def test_header_with_no_sequence_yields_an_empty_string(self, tmp_path):
        path = tmp_path / "x.fa"
        path.write_text(">only\n", encoding="utf-8")
        assert list(common.read_fasta(path)) == [("only", "")]

    def test_consecutive_headers_yield_an_empty_record(self, tmp_path):
        path = tmp_path / "x.fa"
        path.write_text(">a\n>b\nAC\n", encoding="utf-8")
        assert list(common.read_fasta(path)) == [("a", ""), ("b", "AC")]

    def test_text_before_the_first_header_is_discarded(self, tmp_path):
        path = tmp_path / "x.fa"
        path.write_text("junk\nmore junk\n>h\nACD\n", encoding="utf-8")
        assert list(common.read_fasta(path)) == [("h", "ACD")]

    def test_headerless_file_yields_nothing(self, tmp_path):
        path = tmp_path / "x.fa"
        path.write_text("ACDEF\nGHIKL\n", encoding="utf-8")
        assert list(common.read_fasta(path)) == []

    def test_inner_whitespace_inside_a_sequence_line_survives(self, tmp_path):
        """Only the ENDS of a sequence line are stripped.  Pinned, not endorsed:
        an internal space would reach ``aligned_byte_matrix`` as code point 32."""
        path = tmp_path / "x.fa"
        path.write_text(">h\nAC GT\n", encoding="utf-8")
        assert list(common.read_fasta(path)) == [("h", "AC GT")]

    def test_is_lazy_so_a_missing_file_raises_only_on_iteration(self, tmp_path):
        gen = common.read_fasta(tmp_path / "absent.fa")   # no exception yet
        with pytest.raises(FileNotFoundError):
            next(gen)

    def test_gaps_and_stops_are_preserved(self, tmp_path):
        path = tmp_path / "x.fa"
        path.write_text(">h\n--AC*D.\n", encoding="utf-8")
        assert list(common.read_fasta(path)) == [("h", "--AC*D.")]


class TestWriteFasta:

    def test_single_line_by_default(self, tmp_path):
        path = tmp_path / "x.fa"
        common.write_fasta(path, [("h", "ABCDEFG")])
        assert path.read_text(encoding="utf-8") == ">h\nABCDEFG\n"

    def test_line_width_wraps(self, tmp_path):
        path = tmp_path / "x.fa"
        common.write_fasta(path, [("h", "ABCDEFG")], line_width=3)
        assert path.read_text(encoding="utf-8") == ">h\nABC\nDEF\nG\n"

    @pytest.mark.parametrize("line_width", [0, -1, -1000])
    def test_non_positive_line_width_means_one_line(self, tmp_path, line_width):
        path = tmp_path / "x.fa"
        common.write_fasta(path, [("h", "ABCDEFG")], line_width=line_width)
        assert path.read_text(encoding="utf-8") == ">h\nABCDEFG\n"

    def test_creates_missing_parents(self, tmp_path):
        path = tmp_path / "a" / "b" / "x.fa"
        common.write_fasta(path, [("h", "AC")])
        assert path.read_text(encoding="utf-8") == ">h\nAC\n"

    def test_empty_record_list_writes_an_empty_file(self, tmp_path):
        path = tmp_path / "x.fa"
        common.write_fasta(path, [])
        assert path.read_bytes() == b""
        assert common.count_fasta_records(path) == 0

    def test_accepts_a_generator(self, tmp_path):
        path = tmp_path / "x.fa"
        common.write_fasta(path, ((f"h{i}", "AC") for i in range(3)))
        assert common.count_fasta_records(path) == 3

    def test_empty_sequence_still_writes_a_newline(self, tmp_path):
        path = tmp_path / "x.fa"
        common.write_fasta(path, [("h", "")])
        assert path.read_text(encoding="utf-8") == ">h\n\n"

    def test_round_trips_through_the_independent_reader(self, tmp_path):
        records = [(f"seq_{i:02d}", QUERY_PROTEIN[i:] or "A") for i in range(5)]
        path = tmp_path / "x.fa"
        common.write_fasta(path, records, line_width=13)
        assert ref_read_fasta(path) == records
        assert list(common.read_fasta(path)) == records

    def test_does_not_uppercase_on_write(self, tmp_path):
        """Asymmetry worth knowing: the reader uppercases, the writer does not."""
        path = tmp_path / "x.fa"
        common.write_fasta(path, [("h", "acd")])
        assert path.read_text(encoding="utf-8") == ">h\nacd\n"
        assert list(common.read_fasta(path)) == [("h", "ACD")]


class TestFastaHelpers:

    def test_read_fasta_sequences_drops_headers(self, tmp_path):
        path = ref_write_fasta(tmp_path / "x.fa", [("a", "ACD"), ("b", "EFG")])
        assert common.read_fasta_sequences(path) == ["ACD", "EFG"]

    def test_read_fasta_sequences_of_an_empty_file(self, tmp_path):
        path = tmp_path / "x.fa"
        path.write_text("", encoding="utf-8")
        assert common.read_fasta_sequences(path) == []

    def test_read_fasta_sequences_keeps_duplicates_and_order(self, tmp_path):
        path = ref_write_fasta(tmp_path / "x.fa", [("a", "AC"), ("b", "AC"), ("c", "GG")])
        assert common.read_fasta_sequences(path) == ["AC", "AC", "GG"]

    def test_count_fasta_records(self, tmp_path):
        path = ref_write_fasta(tmp_path / "x.fa", [(f"h{i}", "AC") for i in range(17)])
        assert common.count_fasta_records(path) == 17

    def test_count_counts_gt_at_line_start_only(self, tmp_path):
        """A '>' inside a description does not start a record; one at column 0 does."""
        path = tmp_path / "x.fa"
        path.write_text(">a >b >c\nAC\n>d\nGG\n", encoding="utf-8")
        assert common.count_fasta_records(path) == 2

    def test_a_gt_inside_a_sequence_line_is_not_a_record(self, tmp_path):
        """``startswith``, not ``in``: the count must agree with ``read_fasta``.

        The two functions are used interchangeably to report panel depth (one is
        the cheap version of the other), so a disagreement would show up as a
        frequency denominator that does not match the record count.
        """
        path = tmp_path / "x.fa"
        path.write_text(">a\nAC>GT\n  >notaheader\n>b\nGG\n", encoding="utf-8")
        records = list(common.read_fasta(path))
        assert [h for h, _ in records] == ["a", "b"]
        assert records[0][1] == "AC>GT>NOTAHEADER"
        assert common.count_fasta_records(path) == 2 == len(records)

    def test_count_agrees_with_read_fasta(self, tmp_path):
        path = tmp_path / "x.fa"
        path.write_bytes(b">a\nAC\n\n>b\n>c\nGG")
        assert common.count_fasta_records(path) == len(list(common.read_fasta(path))) == 3

    def test_count_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            common.count_fasta_records(tmp_path / "absent.fa")


# =========================================================================== #
# translate_reference_cds
# =========================================================================== #

class TestTranslateReferenceCds:

    def test_query_cds_translates_to_the_query_protein(self):
        assert common.translate_reference_cds(QUERY_CDS) == QUERY_PROTEIN
        assert common.translate_reference_cds(QUERY_CDS) == ref_translate_cds(QUERY_CDS)

    @pytest.mark.parametrize("raw", ["", "A", "AT", "--", "..", "---.."])
    def test_shorter_than_one_codon_gives_empty(self, raw):
        assert common.translate_reference_cds(raw) == ""

    def test_trailing_partial_codon_is_dropped(self):
        assert common.translate_reference_cds("ATGGCTAC") == "MA"
        assert common.translate_reference_cds("ATGGCT") == "MA"

    def test_lowercase_and_uracil_are_normalised(self):
        assert common.translate_reference_cds("auggcu") == "MA"
        assert common.translate_reference_cds("AUGGCU") == "MA"

    def test_gaps_and_dots_are_removed_before_framing(self):
        """Removal happens FIRST, so gaps do not shift the reading frame."""
        assert common.translate_reference_cds("A-TG.GCT") == "MA"
        assert common.translate_reference_cds("ATGGCT") == "MA"

    def test_internal_stops_are_stripped_not_truncated_at(self):
        """The byte-faithful bit: ``.replace('*','')``, NOT ``to_stop=True``.

        ``ATG TGA GCT`` -> 'M', '*', 'A' -> "MA".  A ``to_stop=True`` mirror would
        return "M" and every downstream position would be shifted.
        """
        assert common.translate_reference_cds("ATGTGAGCT") == "MA"
        assert common.translate_reference_cds("ATGTAAGCTTAG") == "MA"

    def test_terminal_stop_is_removed(self):
        assert common.translate_reference_cds("ATGGCTTGA") == "MA"

    def test_ambiguity_codes_translate_to_x(self):
        assert common.translate_reference_cds("ATGNNN") == "MX"

    def test_length_is_floor_div_three_minus_stops(self):
        assert len(common.translate_reference_cds(QUERY_CDS)) == len(QUERY_CDS) // 3 - 1

    def test_is_idempotent_under_repeated_normalisation(self):
        once = common.translate_reference_cds("a-ug.gcu")
        assert once == "MA"
        # Feeding a protein back in is nonsense but must not crash.
        assert isinstance(common.translate_reference_cds(once), str)


# =========================================================================== #
# load_reference_cds
# =========================================================================== #

class TestLoadReferenceCds:

    def test_happy_path(self, query_cds_fasta):
        payload = common.load_reference_cds(query_cds_fasta, label="K")
        assert payload["protein"] == QUERY_PROTEIN
        assert payload["nucleotide"] == QUERY_CDS
        assert payload["lineage"] == "K"
        assert payload["path"] == str(query_cds_fasta)
        assert payload["header"] == "EPI0000001|HA|A/Synthetic/1/2025|EPI_ISL_0000001|K"
        assert set(payload) == {"header", "lineage", "nucleotide", "protein", "path"}

    def test_label_defaults_to_empty(self, query_cds_fasta):
        assert common.load_reference_cds(query_cds_fasta)["lineage"] == ""

    def test_gaps_are_stripped_from_the_stored_nucleotide(self, tmp_path):
        path = ref_write_fasta(tmp_path / "r.fa", [("h", "a-ug.gcu")])
        payload = common.load_reference_cds(path)
        assert payload["nucleotide"] == "ATGGCT"
        assert payload["protein"] == "MA"

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "r.fa"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="No records found in reference FASTA"):
            common.load_reference_cds(path)

    def test_missing_file_raises_filenotfound(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            common.load_reference_cds(tmp_path / "absent.fa")

    def test_multi_record_warns_and_uses_the_first(self, tmp_path, capsys):
        path = ref_write_fasta(
            tmp_path / "r.fa", [("first", "ATGGCT"), ("second", "ATGTGT")]
        )
        payload = common.load_reference_cds(path)
        assert payload["header"] == "first"
        assert payload["protein"] == "MA"
        out = capsys.readouterr().out
        assert "contains 2 records" in out
        assert "using the first only" in out

    def test_single_record_does_not_warn(self, query_cds_fasta, capsys):
        common.load_reference_cds(query_cds_fasta)
        assert capsys.readouterr().out == ""

    def test_protein_input_is_rejected_and_the_message_names_the_characters(self, tmp_path):
        path = ref_write_fasta(tmp_path / "r.fa", [("h", QUERY_PROTEIN)])
        with pytest.raises(ValueError) as excinfo:
            common.load_reference_cds(path)
        message = str(excinfo.value)
        assert "does not look like a nucleotide CDS" in message
        assert str(path) in message
        # Every offending residue must be listed, sorted.
        offenders = sorted(set(QUERY_PROTEIN) - set("ACGTNRYKMSWBDHV"))
        assert str(offenders) in message

    def test_empty_sequence_is_rejected(self, tmp_path):
        path = tmp_path / "r.fa"
        path.write_text(">only-a-header\n", encoding="utf-8")
        with pytest.raises(ValueError, match="does not look like a nucleotide CDS"):
            common.load_reference_cds(path)

    def test_all_gap_sequence_is_rejected(self, tmp_path):
        path = ref_write_fasta(tmp_path / "r.fa", [("h", "-----...")])
        with pytest.raises(ValueError, match="does not look like a nucleotide CDS"):
            common.load_reference_cds(path)

    @pytest.mark.parametrize("code", list("NRYKMSWBDHV"))
    def test_iupac_ambiguity_codes_are_accepted(self, tmp_path, code):
        path = ref_write_fasta(tmp_path / "r.fa", [("h", "ATG" + code * 3)])
        payload = common.load_reference_cds(path)
        assert payload["nucleotide"] == "ATG" + code * 3

    def test_uracil_is_accepted_and_stored_as_thymine(self, tmp_path):
        path = ref_write_fasta(tmp_path / "r.fa", [("h", "AUGGCU")])
        payload = common.load_reference_cds(path)
        assert payload["nucleotide"] == "ATGGCT"
        assert payload["protein"] == "MA"

    def test_header_is_the_full_description_not_the_seqio_id(self, tmp_path):
        """Divergence from ``_load_single_focal_reference``, pinned deliberately.

        The original returns ``record.id`` -- the first whitespace-delimited token.
        This returns the whole description.  Harmless today (it is only recorded
        as ``reference_header`` metadata, and real GISAID HA headers have no
        spaces), but it is NOT the byte-faithful behaviour the module docstring
        advertises, so it is asserted rather than assumed.
        """
        path = ref_write_fasta(tmp_path / "r.fa", [("EPI1 A/England/1/2025", "ATGGCT")])
        assert common.load_reference_cds(path)["header"] == "EPI1 A/England/1/2025"


# =========================================================================== #
# read_guide_rows
# =========================================================================== #

class TestReadGuideRows:

    def test_five_lineage_guide(self, five_lineage_guide):
        rows = common.read_guide_rows(five_lineage_guide["path"])
        assert [row["label"] for row in rows] == list(LINEAGE_ORDER)
        for row, (label, panel, reference) in zip(rows, five_lineage_guide["rows"]):
            assert row == {
                "label": label,
                "diversity_path": str(panel),
                "reference_path": str(reference),
            }

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Guide file not found"):
            common.read_guide_rows(tmp_path / "absent.csv")

    def test_header_only_file_gives_no_rows(self, tmp_path):
        path = tmp_path / "g.csv"
        path.write_text("month,fasta,reference\n", encoding="utf-8")
        assert common.read_guide_rows(path) == []

    def test_completely_empty_file_gives_no_rows(self, tmp_path):
        path = tmp_path / "g.csv"
        path.write_text("", encoding="utf-8")
        assert common.read_guide_rows(path) == []

    def test_label_and_path_column_aliases(self, tmp_path):
        path = tmp_path / "g.csv"
        path.write_text("label,path,reference\nK,/p/k.fa,/r/k.nt\n", encoding="utf-8")
        assert common.read_guide_rows(path) == [
            {"label": "K", "diversity_path": "/p/k.fa", "reference_path": "/r/k.nt"}
        ]

    def test_month_wins_over_label_when_both_present(self, tmp_path):
        path = tmp_path / "g.csv"
        path.write_text("month,label,fasta\nMONTH,LABEL,/p.fa\n", encoding="utf-8")
        assert common.read_guide_rows(path)[0]["label"] == "MONTH"

    def test_empty_month_falls_back_to_label(self, tmp_path):
        path = tmp_path / "g.csv"
        path.write_text("month,label,fasta\n,LABEL,/p.fa\n", encoding="utf-8")
        assert common.read_guide_rows(path)[0]["label"] == "LABEL"

    def test_fasta_wins_over_path_when_both_present(self, tmp_path):
        path = tmp_path / "g.csv"
        path.write_text("month,fasta,path\nK,/fasta.fa,/path.fa\n", encoding="utf-8")
        assert common.read_guide_rows(path)[0]["diversity_path"] == "/fasta.fa"

    def test_reference_is_optional(self, tmp_path):
        path = tmp_path / "g.csv"
        path.write_text("month,fasta\nK,/p/k.fa\n", encoding="utf-8")
        assert common.read_guide_rows(path)[0]["reference_path"] == ""

    def test_rows_without_a_label_or_a_fasta_are_skipped_silently(self, tmp_path):
        path = tmp_path / "g.csv"
        path.write_text(
            "month,fasta,reference\n"
            "K,/p/k.fa,/r/k.nt\n"
            ",/p/x.fa,/r/x.nt\n"      # no label
            "J,,\n"                    # no fasta
            "  ,  ,  \n"               # neither
            "J.2.4,/p/j.fa,\n",
            encoding="utf-8",
        )
        assert [row["label"] for row in common.read_guide_rows(path)] == ["K", "J.2.4"]

    def test_whitespace_is_stripped_from_every_field(self, tmp_path):
        path = tmp_path / "g.csv"
        path.write_text("month,fasta,reference\n  K  , /p/k.fa , /r/k.nt \n", encoding="utf-8")
        assert common.read_guide_rows(path) == [
            {"label": "K", "diversity_path": "/p/k.fa", "reference_path": "/r/k.nt"}
        ]

    def test_dotted_labels_survive_the_csv_round_trip(self, tmp_path):
        path = tmp_path / "g.csv"
        path.write_text(
            "month,fasta\n" + "".join(f"{label},/p/{label}.fa\n" for label in LINEAGE_ORDER),
            encoding="utf-8",
        )
        assert [row["label"] for row in common.read_guide_rows(path)] == list(LINEAGE_ORDER)

    def test_unknown_columns_are_ignored(self, tmp_path):
        path = tmp_path / "g.csv"
        path.write_text("month,fasta,junk\nK,/p/k.fa,whatever\n", encoding="utf-8")
        assert common.read_guide_rows(path)[0]["label"] == "K"

    def test_quoted_fields_with_commas(self, tmp_path):
        path = tmp_path / "g.csv"
        path.write_text('month,fasta\n"A,B",/p/k.fa\n', encoding="utf-8")
        assert common.read_guide_rows(path)[0]["label"] == "A,B"

    def test_a_directory_raises_rather_than_returning_nothing(self, tmp_path):
        target = tmp_path / "adir"
        target.mkdir()
        with pytest.raises(OSError):
            common.read_guide_rows(target)


# =========================================================================== #
# parse_parent_map_override
# =========================================================================== #

class TestParseParentMapOverride:

    @pytest.mark.parametrize(
        "spec, expected",
        [
            (None, {}),
            ("", {}),
            ("   ", {}),
            (",", {}),
            ("  ,  ,  ", {}),
            ("K=J.2_int", {"K": "J.2_int"}),
            (" K = J.2_int ", {"K": "J.2_int"}),
            ("K=J.2_int,J.2.4=J_int", {"K": "J.2_int", "J.2.4": "J_int"}),
            ("K=J.2_int,,J.2.4=J_int", {"K": "J.2_int", "J.2.4": "J_int"}),
            ("K=J=X", {"K": "J=X"}),          # only the FIRST '=' splits
            ("K=A,K=B", {"K": "B"}),           # last wins
        ],
    )
    def test_parse(self, spec, expected):
        assert common.parse_parent_map_override(spec) == expected

    @pytest.mark.parametrize("spec", ["Knope", "K,J=P", "K=P,Q"])
    def test_entry_without_an_equals_raises(self, spec):
        with pytest.raises(ValueError, match="is not of the form child=parent"):
            common.parse_parent_map_override(spec)

    def test_error_message_names_the_offending_chunk_only(self):
        with pytest.raises(ValueError) as excinfo:
            common.parse_parent_map_override("K=P,Qnope")
        assert "'Qnope'" in str(excinfo.value)
        assert "--parent-map" in str(excinfo.value)

    @pytest.mark.parametrize(
        "spec, expected",
        [("K=", {"K": ""}), ("=P", {"": "P"}), ("=", {"": ""})],
    )
    def test_empty_sides_are_accepted_here_and_rejected_downstream(self, spec, expected):
        """``parse_parent_map_override`` is permissive; ``resolve_parent_map`` is not.

        ``constants.parse_edge_spec`` -- which parses the SAME shape of string for
        ``--sensitivity-parent-map`` -- rejects these outright, so the two parsers
        disagree.  Both end up refusing the input, but only one of them says why
        at parse time.
        """
        assert common.parse_parent_map_override(spec) == expected
        with pytest.raises(ValueError, match="malformed edge"):
            constants.parse_edge_spec(spec)

    def test_agrees_with_parse_edge_spec_on_well_formed_input(self):
        spec = "K=J.2_int,J.2.4=J_int"
        assert common.parse_parent_map_override(spec) == constants.parse_edge_spec(spec)


# =========================================================================== #
# resolve_parent_map
# =========================================================================== #

class TestResolveParentMap:

    def test_default_preset_against_the_five_lineage_guide(self, five_lineage_guide):
        resolved = common.resolve_parent_map(
            constants.DEFAULT_PARENT_MAP_PRESET, None, five_lineage_guide["labels"]
        )
        assert resolved == EXPECTED_PARENT_MAP
        assert resolved["K"] == "J.2.4"

    def test_sensitivity_preset(self, five_lineage_guide):
        resolved = common.resolve_parent_map(
            "brief_as_stated", None, five_lineage_guide["labels"]
        )
        assert resolved == EXPECTED_SENSITIVITY_PARENT_MAP
        assert resolved["K"] == "J.2_int"

    def test_the_two_presets_differ_on_exactly_the_contested_edge(self, five_lineage_guide):
        a = common.resolve_parent_map("clade_evidence", None, five_lineage_guide["labels"])
        b = common.resolve_parent_map("brief_as_stated", None, five_lineage_guide["labels"])
        assert {k for k in a if a[k] != b[k]} == {"K"}

    def test_unknown_preset_raises_and_lists_the_known_ones(self, five_lineage_guide):
        with pytest.raises(ValueError) as excinfo:
            common.resolve_parent_map("nope", None, five_lineage_guide["labels"])
        message = str(excinfo.value)
        assert "Unknown parent-map preset 'nope'" in message
        assert "['brief_as_stated', 'clade_evidence']" in message

    def test_preset_is_validated_before_the_labels(self):
        """An unknown preset must not be masked by an empty guide."""
        with pytest.raises(ValueError, match="Unknown parent-map preset"):
            common.resolve_parent_map("nope", None, [])

    def test_override_replaces_a_single_edge(self, five_lineage_guide):
        resolved = common.resolve_parent_map(
            "clade_evidence", "K=J.2_int", five_lineage_guide["labels"]
        )
        assert resolved["K"] == "J.2_int"
        assert {k: v for k, v in resolved.items() if k != "K"} == {
            k: v for k, v in EXPECTED_PARENT_MAP.items() if k != "K"
        }

    def test_override_can_add_a_new_edge(self, guide_factory):
        guide = guide_factory(list(LINEAGE_ORDER) + ["L.1"])
        resolved = common.resolve_parent_map("clade_evidence", "L.1=K", guide["labels"])
        assert resolved["L.1"] == "K"
        assert len(resolved) == len(EXPECTED_PARENT_MAP) + 1

    def test_empty_override_is_a_no_op(self, five_lineage_guide):
        for override in (None, "", "   "):
            assert common.resolve_parent_map(
                "clade_evidence", override, five_lineage_guide["labels"]
            ) == EXPECTED_PARENT_MAP

    def test_result_is_a_copy_so_mutating_it_cannot_corrupt_the_preset(self, five_lineage_guide):
        resolved = common.resolve_parent_map(
            "clade_evidence", None, five_lineage_guide["labels"]
        )
        resolved["K"] = "CORRUPTED"
        assert constants.DEFAULT_PARENT_MAPS["clade_evidence"]["K"] == "J.2.4"
        assert common.PARENT_MAP_PRESETS["clade_evidence"]["K"] == "J.2.4"

    def test_unknown_child_raises(self, five_lineage_guide):
        with pytest.raises(ValueError) as excinfo:
            common.resolve_parent_map(
                "clade_evidence", "Z.9=K", five_lineage_guide["labels"]
            )
        assert "names child 'Z.9'" in str(excinfo.value)
        assert "no guide row" in str(excinfo.value)

    def test_unknown_parent_raises_and_names_both_ends(self, five_lineage_guide):
        with pytest.raises(ValueError) as excinfo:
            common.resolve_parent_map(
                "clade_evidence", "K=Z.9", five_lineage_guide["labels"]
            )
        message = str(excinfo.value)
        assert "names parent 'Z.9'" in message
        assert "(of 'K')" in message

    def test_empty_parent_from_a_trailing_equals_is_caught(self, five_lineage_guide):
        with pytest.raises(ValueError, match=r"names parent ''"):
            common.resolve_parent_map("clade_evidence", "K=", five_lineage_guide["labels"])

    def test_empty_child_from_a_leading_equals_is_caught(self, five_lineage_guide):
        with pytest.raises(ValueError, match=r"names child ''"):
            common.resolve_parent_map("clade_evidence", "=K", five_lineage_guide["labels"])

    def test_empty_guide_rejects_the_first_preset_edge(self):
        with pytest.raises(ValueError, match="names child 'J_int'"):
            common.resolve_parent_map("clade_evidence", None, [])

    def test_self_parent_raises(self, five_lineage_guide):
        with pytest.raises(ValueError, match=r"makes 'K' its own parent"):
            common.resolve_parent_map("clade_evidence", "K=K", five_lineage_guide["labels"])

    def test_two_node_cycle_raises(self, five_lineage_guide):
        with pytest.raises(ValueError, match="contains a cycle"):
            common.resolve_parent_map(
                "clade_evidence", "J.2.4=K", five_lineage_guide["labels"]
            )

    def test_three_node_cycle_raises(self, five_lineage_guide):
        # J_int -> J.2.4 -> J.2_int -> J_int
        with pytest.raises(ValueError, match="contains a cycle"):
            common.resolve_parent_map(
                "clade_evidence", "J_int=J.2.4", five_lineage_guide["labels"]
            )

    def test_full_cycle_through_every_node_raises(self, five_lineage_guide):
        with pytest.raises(ValueError, match="contains a cycle"):
            common.resolve_parent_map(
                "clade_evidence", "G.1=K", five_lineage_guide["labels"]
            )

    def test_cycle_message_names_a_node_on_the_cycle(self, five_lineage_guide):
        with pytest.raises(ValueError) as excinfo:
            common.resolve_parent_map(
                "clade_evidence", "J.2.4=K", five_lineage_guide["labels"]
            )
        assert "'K'" in str(excinfo.value) or "'J.2.4'" in str(excinfo.value)

    def test_cycle_check_terminates_on_a_disconnected_cycle(self, guide_factory):
        """A cycle that no preset edge touches must still be found and must not hang.

        The preset edges stay valid (the five real labels are in the guide); the
        A->B->C->A cycle added by the override shares no node with them, so the
        walk that finds it has to start from a node the preset never mentions.
        """
        guide = guide_factory(list(LINEAGE_ORDER) + ["A", "B", "C"])
        with pytest.raises(ValueError, match="contains a cycle"):
            common.resolve_parent_map(
                "clade_evidence", "A=B,B=C,C=A", guide["labels"]
            )

    def test_a_valid_forest_with_two_roots_is_accepted(self, guide_factory):
        guide = guide_factory(list(LINEAGE_ORDER) + ["A", "B", "C", "D"])
        resolved = common.resolve_parent_map("clade_evidence", "A=B,C=D", guide["labels"])
        assert resolved == {**EXPECTED_PARENT_MAP, "A": "B", "C": "D"}

    def test_a_diamond_is_accepted_because_only_cycles_are_refused(self, guide_factory):
        """Two children sharing a parent is legal; the check is acyclicity, not a tree."""
        guide = guide_factory(list(LINEAGE_ORDER) + ["A", "B", "C"])
        assert common.resolve_parent_map("clade_evidence", "A=C,B=C", guide["labels"]) == {
            **EXPECTED_PARENT_MAP, "A": "C", "B": "C"
        }

    def test_known_labels_may_be_any_sequence(self, five_lineage_guide):
        as_tuple = common.resolve_parent_map(
            "clade_evidence", None, tuple(five_lineage_guide["labels"])
        )
        as_set = common.resolve_parent_map(
            "clade_evidence", None, set(five_lineage_guide["labels"])
        )
        assert as_tuple == as_set == EXPECTED_PARENT_MAP

    def test_duplicate_guide_labels_are_harmless(self, five_lineage_guide):
        labels = list(five_lineage_guide["labels"]) * 2
        assert common.resolve_parent_map("clade_evidence", None, labels) == EXPECTED_PARENT_MAP

    def test_child_check_precedes_the_parent_check(self, five_lineage_guide):
        """Both ends unknown -> the CHILD is reported, so the message is stable."""
        with pytest.raises(ValueError, match="names child 'Z.1'"):
            common.resolve_parent_map(
                "clade_evidence", "Z.1=Z.2", five_lineage_guide["labels"]
            )


# =========================================================================== #
# constants naming helpers
# =========================================================================== #

class TestVariantParentToken:

    @pytest.mark.parametrize(
        "parent, token",
        [
            ("G.1", "G1"),
            ("J_int", "Jint"),
            ("J.2_int", "J2int"),
            ("J.2.4", "J24"),
            ("K", "K"),
        ],
    )
    def test_pinned_tokens(self, parent, token):
        assert constants.variant_parent_token(parent) == token
        assert common.variant_parent_token(parent) == token

    def test_tokens_are_injective_over_the_real_label_set(self):
        tokens = [constants.variant_parent_token(label) for label in LINEAGE_ORDER]
        assert sorted(tokens) == ["G1", "J24", "J2int", "Jint", "K"]
        assert len(set(tokens)) == len(LINEAGE_ORDER)

    def test_tokens_are_alphanumeric_so_they_survive_a_variant_name(self):
        for label in LINEAGE_ORDER:
            token = constants.variant_parent_token(label)
            assert token.isalnum()
            name = f"PRESCOTT_eq2_c0p50_k1_parent{token}"
            assert name.rsplit("_parent", 1)[1] == token

    def test_token_is_not_injective_in_general(self):
        """``J.2.4`` and ``J24`` collapse; so do ``J.2_int`` and ``J2int``.

        Only the pinned five-label set is safe, which is why the driver reads the
        parent back out of ``score_variants.tsv`` rather than out of the name.
        """
        assert constants.variant_parent_token("J.2.4") == constants.variant_parent_token("J24")
        assert constants.variant_parent_token("J.2_int") == constants.variant_parent_token("J2int")

    def test_non_string_input_is_coerced(self):
        assert constants.variant_parent_token(2.4) == "24"

    @pytest.mark.parametrize("degenerate", ["", "_", ".", "._.", "___"])
    def test_degenerate_labels_produce_an_empty_token(self, degenerate):
        assert constants.variant_parent_token(degenerate) == ""


class TestAlternateFrequencyBasename:

    def test_pinned_names(self):
        assert constants.alternate_frequency_basename("K", "J.2_int") == "K_parentJ2int_frequency"
        assert constants.alternate_frequency_basename("K", "J.2.4") == "K_parentJ24_frequency"
        assert (
            constants.alternate_frequency_basename("J.2_int", "J_int")
            == "J.2_int_parentJint_frequency"
        )

    def test_the_lineage_key_keeps_its_dots(self):
        """The key is a ``safe_label`` key, not a ``dot_free_key``, so dots stay."""
        name = constants.alternate_frequency_basename("J.2.4", "J.2_int")
        assert name.startswith("J.2.4_parent")
        assert Path(name + ".txt").suffix == ".txt"

    @pytest.mark.parametrize("child", LINEAGE_ORDER)
    @pytest.mark.parametrize("parent", LINEAGE_ORDER)
    def test_never_collides_with_the_primary_name_for_real_labels(self, child, parent):
        primary = f"{child}_parent_frequency"
        assert constants.alternate_frequency_basename(child, parent) != primary

    def test_alternate_names_are_injective_over_the_real_label_set(self):
        names = [
            constants.alternate_frequency_basename(child, parent)
            for child in LINEAGE_ORDER
            for parent in LINEAGE_ORDER
        ]
        assert len(set(names)) == len(names)

    def test_an_empty_parent_label_DOES_collide_with_the_primary(self):
        """Latent hazard, pinned so it cannot appear by accident.

        The docstring claims "the two forms cannot collide".  They do when the
        token is empty -- an empty, dots-only or underscores-only parent label
        regenerates the PRIMARY basename and would overwrite it.  Unreachable
        through the CLI today (``prepare_inputs`` validates the sensitivity parent
        against the guide labels first), so this is an API contract gap rather
        than a live bug -- but it is exactly one missing validation away.
        """
        for degenerate in ("", "_", "."):
            assert constants.alternate_frequency_basename("K", degenerate) == "K_parent_frequency"


class TestSensitivityEdgesBetweenPresets:

    def test_default_vs_implicit_other(self):
        assert constants.sensitivity_edges_between_presets("clade_evidence") == {"K": "J.2_int"}

    def test_reverse_direction(self):
        assert constants.sensitivity_edges_between_presets("brief_as_stated") == {"K": "J.2.4"}

    def test_explicit_other_preset(self):
        assert constants.sensitivity_edges_between_presets(
            "clade_evidence", "brief_as_stated"
        ) == {"K": "J.2_int"}

    def test_a_preset_against_itself_has_no_disagreements(self):
        for name in constants.preset_names():
            assert constants.sensitivity_edges_between_presets(name, name) == {}

    def test_unknown_primary_preset_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unknown parent-map preset 'nope'"):
            constants.sensitivity_edges_between_presets("nope")

    def test_unknown_other_preset_raises_keyerror_not_valueerror(self):
        """Asymmetric validation, pinned: only the FIRST argument gets a nice error."""
        with pytest.raises(KeyError):
            constants.sensitivity_edges_between_presets("clade_evidence", "nope")

    def test_single_preset_registry_yields_no_edges(self, monkeypatch):
        monkeypatch.setattr(
            constants, "DEFAULT_PARENT_MAPS", {"only": dict(EXPECTED_PARENT_MAP)}
        )
        assert constants.sensitivity_edges_between_presets("only") == {}

    def test_child_absent_from_the_other_preset_is_not_an_edge(self, monkeypatch):
        monkeypatch.setattr(
            constants,
            "DEFAULT_PARENT_MAPS",
            {
                "a": {"K": "J.2.4", "L": "K"},
                "b": {"K": "J.2.4"},
            },
        )
        assert constants.sensitivity_edges_between_presets("a", "b") == {}


class TestParseEdgeSpec:

    @pytest.mark.parametrize(
        "spec, expected",
        [
            ("K=J.2_int", {"K": "J.2_int"}),
            ("K=J.2_int,J.2.4=J_int", {"K": "J.2_int", "J.2.4": "J_int"}),
            (" K = J.2_int , ", {"K": "J.2_int"}),
            ("", {}),
            ("   ", {}),
            (",,", {}),
            (None, {}),
            ("K=A,K=B", {"K": "B"}),
            ("K=A=B", {"K": "A=B"}),
        ],
    )
    def test_parse(self, spec, expected):
        assert constants.parse_edge_spec(spec) == expected

    @pytest.mark.parametrize("spec", ["K", "K=", "=P", "=", " = ", "K=P,Q"])
    def test_malformed_edges_raise(self, spec):
        with pytest.raises(ValueError, match="malformed edge"):
            constants.parse_edge_spec(spec)

    def test_error_message_quotes_the_offending_token(self):
        with pytest.raises(ValueError) as excinfo:
            constants.parse_edge_spec("K=J.2_int,Qnope")
        assert "'Qnope'" in str(excinfo.value)

    def test_dotted_labels_survive_both_sides(self):
        assert constants.parse_edge_spec("J.2.4=J.2_int") == {"J.2.4": "J.2_int"}


# =========================================================================== #
# aligned_byte_matrix
# =========================================================================== #

class TestAlignedByteMatrix:

    def test_empty_input(self):
        matrix, aln_len = common.aligned_byte_matrix([])
        assert matrix.shape == (0, 0)
        assert matrix.dtype == np.uint8
        assert aln_len == 0

    def test_single_empty_sequence(self):
        matrix, aln_len = common.aligned_byte_matrix([""])
        assert matrix.shape == (1, 0)
        assert aln_len == 0

    def test_ascii_codes_are_exact(self):
        matrix, aln_len = common.aligned_byte_matrix(["ACD", "EFG"])
        assert aln_len == 3
        assert matrix.tolist() == [[65, 67, 68], [69, 70, 71]]

    def test_short_rows_are_right_padded_with_gaps(self):
        matrix, aln_len = common.aligned_byte_matrix(["ACD", "AC", ""])
        assert aln_len == 3
        assert matrix.tolist() == [
            [65, 67, 68],
            [65, 67, ord("-")],
            [ord("-")] * 3,
        ]

    def test_shape_is_n_seq_by_aln_len(self):
        rows = [QUERY_PROTEIN] * 7
        matrix, aln_len = common.aligned_byte_matrix(rows)
        assert matrix.shape == (7, len(QUERY_PROTEIN))
        assert aln_len == len(QUERY_PROTEIN)

    def test_non_ascii_becomes_a_single_question_mark_byte(self):
        """Length must be preserved, or every downstream column index shifts."""
        matrix, aln_len = common.aligned_byte_matrix(["AÅD"])
        assert aln_len == 3
        assert matrix.tolist() == [[65, ord("?"), 68]]

    def test_uint8_ordering_is_alphabetical_for_uppercase(self):
        """The claim the consensus tie-break rests on."""
        codes = [ord(aa) for aa in AA20]
        assert codes == sorted(codes)

    def test_rows_are_independent(self):
        matrix, _ = common.aligned_byte_matrix(["AAA", "CCC"])
        assert matrix[0].tolist() == [65, 65, 65]
        assert matrix[1].tolist() == [67, 67, 67]


# =========================================================================== #
# build_consensus_and_column_map
# =========================================================================== #

class TestBuildConsensusAndColumnMap:

    def test_empty_input(self):
        assert common.build_consensus_and_column_map([]) == ("", [], 0)

    def test_all_empty_sequences(self):
        assert common.build_consensus_and_column_map(["", "", ""]) == ("", [], 0)

    def test_identical_rows_reproduce_the_sequence(self):
        consensus, cols, aln_len = common.build_consensus_and_column_map([QUERY_PROTEIN] * 4)
        assert consensus == QUERY_PROTEIN
        assert cols == list(range(1, len(QUERY_PROTEIN) + 1))
        assert aln_len == len(QUERY_PROTEIN)

    def test_columns_are_one_based(self):
        consensus, cols, aln_len = common.build_consensus_and_column_map(["AC"])
        assert (consensus, cols, aln_len) == ("AC", [1, 2], 2)

    def test_all_gap_columns_are_dropped_but_still_counted_in_aln_len(self):
        consensus, cols, aln_len = common.build_consensus_and_column_map(["--ACD"] * 3)
        assert consensus == "ACD"
        assert cols == [3, 4, 5]
        assert aln_len == 5

    def test_interior_gap_column_is_dropped(self):
        consensus, cols, aln_len = common.build_consensus_and_column_map(["A-C", "A-C"])
        assert (consensus, cols, aln_len) == ("AC", [1, 3], 3)

    @pytest.mark.parametrize("char", ["-", "*", "."])
    def test_each_ignore_char_makes_a_column_empty(self, char):
        consensus, cols, _ = common.build_consensus_and_column_map([f"A{char}C"] * 3)
        assert (consensus, cols) == ("AC", [1, 3])

    def test_majority_wins(self):
        consensus, _, _ = common.build_consensus_and_column_map(["A", "A", "C"])
        assert consensus == "A"
        consensus, _, _ = common.build_consensus_and_column_map(["C", "C", "A"])
        assert consensus == "C"

    def test_ties_break_alphabetically(self):
        """2 x A vs 2 x C -> 'A'; 2 x W vs 2 x Y -> 'W'.  Mirrors np.unique+argmax."""
        assert common.build_consensus_and_column_map(["A", "C", "A", "C"])[0] == "A"
        assert common.build_consensus_and_column_map(["C", "A", "C", "A"])[0] == "A"
        assert common.build_consensus_and_column_map(["Y", "W", "Y", "W"])[0] == "W"

    def test_tie_break_is_independent_of_row_order(self):
        import itertools
        for perm in itertools.permutations("AACC"):
            assert common.build_consensus_and_column_map(list(perm))[0] == "A"

    def test_gaps_do_not_win_a_column_they_dominate(self):
        """11 gaps and 1 residue: the residue is the consensus, not the gap."""
        rows = ["-"] * 11 + ["W"]
        assert common.build_consensus_and_column_map(rows)[0] == "W"

    def test_lowercase_residues_are_not_valid(self):
        """``read_fasta`` uppercases, so this only bites a hand-built input."""
        consensus, cols, aln_len = common.build_consensus_and_column_map(["acd"])
        assert (consensus, cols, aln_len) == ("", [], 3)

    def test_custom_valid_residues_restrict_the_alphabet(self):
        consensus, cols, aln_len = common.build_consensus_and_column_map(
            ["AXC"], valid_residues="AC"
        )
        assert (consensus, cols, aln_len) == ("AC", [1, 3], 3)

    def test_custom_ignore_chars_subtract_from_valid_residues(self):
        consensus, cols, _ = common.build_consensus_and_column_map(
            ["AAC", "AAC", "XXC"], ignore_chars=set("-*.C")
        )
        assert (consensus, cols) == ("AA", [1, 2])

    def test_ragged_rows_are_padded_before_counting(self):
        consensus, cols, aln_len = common.build_consensus_and_column_map(["ACD", "AC"])
        assert (consensus, cols, aln_len) == ("ACD", [1, 2, 3], 3)

    def test_column_list_length_always_matches_the_consensus(self, tiny_msa):
        consensus, cols, aln_len = common.build_consensus_and_column_map(tiny_msa["rows"])
        assert len(consensus) == len(cols)
        assert aln_len == tiny_msa["n_columns"]

    def test_tiny_msa_consensus_matches_the_fixture_ground_truth(self, tiny_msa):
        """Column classes computed by ``conftest`` from the literal rows, not by us."""
        consensus, cols, _ = common.build_consensus_and_column_map(tiny_msa["rows"])
        counts = tiny_msa["column_counts"]
        by_column = dict(zip(cols, consensus))
        for pos, expected_counts in counts.items():
            residues = {c: n for c, n in expected_counts.items() if c not in EXPECTED_IGNORE_CHARS}
            if not residues:
                assert pos not in by_column
                continue
            best = max(sorted(residues), key=lambda c: residues[c])
            assert by_column[pos] == best

    def test_all_gap_columns_of_the_tiny_msa_still_survive(self, tiny_msa):
        """They are all-gap-EXCEPT-the-query, so the query residue is the consensus."""
        consensus, cols, _ = common.build_consensus_and_column_map(tiny_msa["rows"])
        by_column = dict(zip(cols, consensus))
        for pos in tiny_msa["all_gap_positions"]:
            assert by_column[pos] == QUERY_PROTEIN[pos - 1]

    def test_a_column_of_only_ignore_chars_across_all_rows_is_dropped(self):
        rows = ["A-C", "A.C", "A*C"]
        consensus, cols, _ = common.build_consensus_and_column_map(rows)
        assert (consensus, cols) == ("AC", [1, 3])


# =========================================================================== #
# map_reference_to_alignment_columns
# =========================================================================== #

class TestMapReferenceToAlignmentColumns:

    def test_identity_mapping(self):
        mapping, aln_len, matched, consensus = common.map_reference_to_alignment_columns(
            "ACD", ["ACD"] * 3
        )
        assert mapping == {1: 1, 2: 2, 3: 3}
        assert (aln_len, matched, consensus) == (3, 3, "ACD")

    def test_leading_gap_columns_shift_the_targets(self):
        mapping, aln_len, matched, consensus = common.map_reference_to_alignment_columns(
            "ACD", ["--ACD"] * 3
        )
        assert mapping == {1: 3, 2: 4, 3: 5}
        assert (aln_len, matched, consensus) == (5, 3, "ACD")

    def test_extra_n_terminal_reference_residue_is_unmapped(self):
        mapping, _, matched, _ = common.map_reference_to_alignment_columns(
            "MACD", ["--ACD"] * 3
        )
        assert mapping == {2: 3, 3: 4, 4: 5}
        assert 1 not in mapping
        assert matched == 3

    def test_reference_deletion_skips_the_consensus_column(self):
        mapping, _, matched, _ = common.map_reference_to_alignment_columns(
            "AD", ["--ACD"] * 3
        )
        assert mapping == {1: 3, 2: 5}
        assert matched == 2

    def test_empty_consensus_returns_early(self):
        """No valid residue anywhere -> empty mapping, empty consensus, real aln_len."""
        mapping, aln_len, matched, consensus = common.map_reference_to_alignment_columns(
            "ACD", ["---", "---"]
        )
        assert (mapping, aln_len, matched, consensus) == ({}, 3, 0, "")

    def test_no_sequences_at_all(self):
        assert common.map_reference_to_alignment_columns("ACD", []) == ({}, 0, 0, "")

    def test_empty_reference_hits_the_no_alignment_branch(self):
        """``globalms('', 'ACDEF')`` returns an EMPTY list, not a gapped alignment.

        This is the only way to reach ``len(alignments) == 0``; the consensus is
        still reported so the caller can log what it failed to map onto.
        """
        from Bio import pairwise2
        assert pairwise2.align.globalms(
            "", "ACDEF", 2.0, -1.0, -10.0, -0.5, one_alignment_only=True
        ) == []
        mapping, aln_len, matched, consensus = common.map_reference_to_alignment_columns(
            "", ["ACDEF"]
        )
        assert (mapping, aln_len, matched, consensus) == ({}, 5, 0, "ACDEF")

    def test_mapping_is_strictly_increasing(self, tiny_msa):
        mapping, _, _, _ = common.map_reference_to_alignment_columns(
            QUERY_PROTEIN, tiny_msa["rows"]
        )
        keys = sorted(mapping)
        assert [mapping[k] for k in keys] == sorted(mapping[k] for k in keys)

    def test_ungapped_msa_maps_every_reference_position_to_itself(self, tiny_msa):
        mapping, aln_len, matched, consensus = common.map_reference_to_alignment_columns(
            QUERY_PROTEIN, tiny_msa["rows"]
        )
        assert aln_len == len(QUERY_PROTEIN)
        assert matched == len(QUERY_PROTEIN)
        assert mapping == {i: i for i in range(1, len(QUERY_PROTEIN) + 1)}
        assert len(consensus) == len(QUERY_PROTEIN)

    def test_matched_pairs_equals_len_mapping_when_positions_are_unique(self):
        mapping, _, matched, _ = common.map_reference_to_alignment_columns(
            QUERY_PROTEIN, [QUERY_PROTEIN] * 3
        )
        assert matched == len(mapping) == len(QUERY_PROTEIN)

    def test_reference_positions_are_one_based(self):
        mapping, _, _, _ = common.map_reference_to_alignment_columns("ACD", ["ACD"])
        assert min(mapping) == 1
        assert 0 not in mapping

    def test_custom_alphabet_is_forwarded_to_the_consensus(self):
        mapping, aln_len, matched, consensus = common.map_reference_to_alignment_columns(
            "AC", ["AXC"], valid_residues="AC"
        )
        assert consensus == "AC"
        assert mapping == {1: 1, 2: 3}
        assert (aln_len, matched) == (3, 2)

    def test_the_cons_pos_bounds_guard_can_never_fire(self):
        """``if 1 <= cons_pos <= len(consensus_to_alignment_col)`` is dead by construction.

        ``build_consensus_and_column_map`` appends to both lists in lockstep, so
        ``len(consensus_seq) == len(consensus_to_alignment_col)`` always, and
        ``cons_pos`` only advances on a non-gap of the consensus row.  The False
        arm is therefore unreachable -- it is retained because the original in
        ``Functions_HuggingFace`` has it, and this test pins the invariant that
        makes it dead so a future edit cannot quietly resurrect it.
        """
        for rows in (["--ACD"] * 3, [QUERY_PROTEIN] * 2, ["A-C", "AGC"], ["ACD", "AC"]):
            consensus, cols, _ = common.build_consensus_and_column_map(rows)
            assert len(consensus) == len(cols)

    def test_completely_unrelated_reference_still_returns_a_mapping(self):
        """globalms always aligns something; the caller checks ``matched_pairs``."""
        mapping, _, matched, _ = common.map_reference_to_alignment_columns(
            "WWWWW", ["AAAAA"] * 3
        )
        assert matched == len(mapping)
        assert matched <= 5


# =========================================================================== #
# Round-tripping lineage labels through the whole naming stack
# =========================================================================== #

class TestLineageNamingRoundTrips:
    """The quirky cases: dotted labels through filenames and back, and collisions."""

    @pytest.mark.parametrize("label", LINEAGE_ORDER)
    def test_every_naming_transform_is_defined_for_every_real_label(self, label):
        assert common.safe_label(label)
        assert common.dot_free_key(label)
        assert common.lineage_tag(label)
        assert constants.variant_parent_token(label)

    def test_the_four_key_flavours_disagree_and_that_is_the_point(self):
        label = "J.2_int"
        assert common.safe_label(label) == "J.2_int"        # filenames: dots survive
        assert common.dot_free_key(label) == "J_2_int"      # prescott -o: no dots
        assert common.lineage_tag(label) == "J2"            # escott prot token
        assert constants.variant_parent_token(label) == "J2int"   # variant suffix
        assert len({"J.2_int", "J_2_int", "J2", "J2int"}) == 4

    @pytest.mark.parametrize("label", LINEAGE_ORDER)
    def test_frequency_filename_round_trip(self, label):
        """``<key>_parent_frequency.txt`` must give the key back, dots and all."""
        key = common.safe_label(label)
        name = f"{key}_parent_frequency.txt"
        assert Path(name).suffix == ".txt"
        assert name[: -len("_parent_frequency.txt")] == key == label

    @pytest.mark.parametrize("label", LINEAGE_ORDER)
    def test_score_matrix_filename_round_trip(self, label):
        key = common.safe_label(label)
        variant = "PRESCOTT_eq2_c0p50_k1_parentJ24"
        name = f"{key}_{variant}_score_matrix.csv"
        assert Path(name).suffix == ".csv"
        stem = name[: -len("_score_matrix.csv")]
        assert stem == f"{key}_{variant}"
        assert stem[: len(key)] == label

    def test_a_dotted_key_would_be_destroyed_by_splitext_but_a_variant_suffix_is_not(self):
        """``PRESCOTT_..._parentJ24`` has no dots, so ``splitext`` is a no-op on it.

        Prefixing it with a dotted lineage key is not: ``splitext`` cuts at the
        LAST dot, so ``J.2.4_PRESCOTT_...`` loses everything from ``.4`` onwards.
        """
        variant = "PRESCOTT_eq2_c0p50_k1_parentJ24"
        assert os.path.splitext(variant) == (variant, "")
        assert os.path.splitext("J.2.4_" + variant) == ("J.2", ".4_" + variant)
        assert os.path.splitext(common.dot_free_key("J.2.4") + "_" + variant) == (
            "J_2_4_" + variant, ""
        )

    def test_the_full_output_basename_prescott_sees_is_dot_free(self):
        for label in LINEAGE_ORDER:
            basename = f"{common.dot_free_key(label)}_HA{common.lineage_tag(label)}"
            assert "." not in basename
            assert os.path.splitext(basename) == (basename, "")

    def test_all_five_lineages_have_pairwise_distinct_names_at_every_level(self):
        for transform in (
            common.safe_label,
            common.dot_free_key,
            common.lineage_tag,
            constants.variant_parent_token,
        ):
            names = [transform(label) for label in LINEAGE_ORDER]
            assert len(set(names)) == len(LINEAGE_ORDER), transform.__name__

    def test_a_j24_lineage_would_collide_with_j_2_4_at_two_levels(self):
        """The one label that must never be added without re-pinning the tags."""
        assert common.lineage_tag("J24") == common.lineage_tag("J.2.4")
        assert constants.variant_parent_token("J24") == constants.variant_parent_token("J.2.4")
        # but NOT at the safe_label / dot_free_key level:
        assert common.safe_label("J24") != common.safe_label("J.2.4")
        assert common.dot_free_key("J24") != common.dot_free_key("J.2.4")


# =========================================================================== #
# Single-sequence and degenerate inputs, end to end through the alignment path
# =========================================================================== #

class TestDegenerateInputs:

    def test_single_sequence_alignment_is_its_own_consensus(self):
        consensus, cols, aln_len = common.build_consensus_and_column_map([QUERY_PROTEIN])
        assert consensus == QUERY_PROTEIN
        assert cols == list(range(1, len(QUERY_PROTEIN) + 1))
        assert aln_len == len(QUERY_PROTEIN)

    def test_single_sequence_maps_to_itself(self):
        mapping, aln_len, matched, consensus = common.map_reference_to_alignment_columns(
            QUERY_PROTEIN, [QUERY_PROTEIN]
        )
        assert mapping == {i: i for i in range(1, len(QUERY_PROTEIN) + 1)}
        assert (aln_len, matched, consensus) == (
            len(QUERY_PROTEIN), len(QUERY_PROTEIN), QUERY_PROTEIN
        )

    def test_single_record_fasta_end_to_end(self, tmp_path):
        path = ref_write_fasta(tmp_path / "one.fa", [(QUERY_HEADER, QUERY_PROTEIN)])
        assert common.count_fasta_records(path) == 1
        sequences = common.read_fasta_sequences(path)
        assert sequences == [QUERY_PROTEIN]
        consensus, _, _ = common.build_consensus_and_column_map(sequences)
        assert consensus == QUERY_PROTEIN

    def test_empty_fasta_end_to_end(self, tmp_path):
        path = tmp_path / "none.fa"
        path.write_text("", encoding="utf-8")
        sequences = common.read_fasta_sequences(path)
        assert sequences == []
        assert common.build_consensus_and_column_map(sequences) == ("", [], 0)
        assert common.map_reference_to_alignment_columns(QUERY_PROTEIN, sequences) == (
            {}, 0, 0, ""
        )

    def test_a_panel_of_only_gaps_produces_no_consensus(self, tmp_path):
        path = ref_write_fasta(tmp_path / "gaps.fa", [(f"h{i}", "-" * 10) for i in range(5)])
        sequences = common.read_fasta_sequences(path)
        assert len(sequences) == 5
        assert common.build_consensus_and_column_map(sequences) == ("", [], 10)


# =========================================================================== #
# Mirror fidelity against Functions_HuggingFace
# =========================================================================== #

@pytest.mark.requires_rma
@pytest.mark.requires_torch
@pytest.mark.integration
class TestMirrorFidelityAgainstRMA:
    """``common`` claims byte-faithful mirrors.  These import the originals and check.

    The mirrors matter because the driver asserts
    ``score_matrix_source_sequence == lineage_data['full_ref_protein']``, and the
    frequency prior has to sit on the same reference->column map that the
    evaluation half uses.  A one-residue drift here is invisible until a figure.
    """

    @staticmethod
    def _rma():
        import Functions_HuggingFace as fh
        return fh

    @staticmethod
    def _records(rows):
        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord
        return [SeqRecord(Seq(row), id=f"r{i}") for i, row in enumerate(rows)]

    AA_TO_CODONS = {aa: [] for aa in AA20}
    IGNORE = {"-", "*", "."}

    @pytest.mark.parametrize(
        "label",
        ["K", "J.2_int", "J.2.4", " K ", "A/H3N2", "a b", "", "A  B/C"],
    )
    def test_safe_label_is_byte_identical(self, label):
        assert common.safe_label(label) == self._rma()._safe_label(label)

    @pytest.mark.parametrize(
        "cds",
        [
            QUERY_CDS,
            "ATGGCT",
            "ATGTGAGCT",
            "auggcu",
            "A-TG.GCT",
            "ATGGCTAC",
            "ATGNNN",
            "",
            "AT",
        ],
    )
    def test_translate_is_byte_identical(self, cds):
        assert common.translate_reference_cds(cds) == self._rma()._translate_nt_to_protein(cds)

    @pytest.mark.parametrize(
        "rows",
        [
            ["--ACDEFG", "--ACDXFG", "--ACDEF-", "MMACDEFG"],
            [QUERY_PROTEIN] * 3,
            ["A", "C", "A", "C"],
            ["---", "---"],
            ["ACD", "AC"],
            [""],
        ],
    )
    def test_consensus_is_byte_identical(self, rows):
        theirs = self._rma()._build_lineage_consensus_and_column_map(
            self._records(rows), self.AA_TO_CODONS, self.IGNORE
        )
        ours = common.build_consensus_and_column_map(rows)
        assert ours == theirs

    @pytest.mark.parametrize(
        "reference, rows",
        [
            ("ACDEFG", ["--ACDEFG", "--ACDXFG", "--ACDEF-", "MMACDEFG"]),
            (QUERY_PROTEIN, [QUERY_PROTEIN] * 4),
            ("MACD", ["--ACD"] * 3),
            ("AD", ["--ACD"] * 3),
            ("ACD", ["---", "---"]),
        ],
    )
    def test_column_map_is_byte_identical(self, reference, rows):
        theirs = self._rma().build_reference_to_alignment_column_map(
            reference, self._records(rows), self.AA_TO_CODONS, self.IGNORE
        )
        mapping, aln_len, matched, _consensus = common.map_reference_to_alignment_columns(
            reference, rows
        )
        assert (mapping, aln_len, matched) == theirs

    def test_randomised_differential_over_120_alignments(self):
        """Seeded, so a failure is reproducible; 120 draws over gaps, ties and lengths."""
        rma = self._rma()
        rng = random.Random(20260805)
        alphabet = AA20 + "-" * 6 + "*."
        for _ in range(120):
            n_rows = rng.randint(1, 6)
            width = rng.randint(1, 14)
            rows = [
                "".join(rng.choice(alphabet) for _ in range(width))
                for _ in range(n_rows)
            ]
            # occasionally make a row short, to exercise the ljust padding
            if rng.random() < 0.3 and width > 2:
                rows[-1] = rows[-1][: rng.randint(1, width - 1)]
            records = self._records(rows)
            assert common.build_consensus_and_column_map(rows) == \
                rma._build_lineage_consensus_and_column_map(
                    records, self.AA_TO_CODONS, self.IGNORE
                )
            reference = "".join(rng.choice(AA20) for _ in range(rng.randint(1, 12)))
            mapping, aln_len, matched, _ = common.map_reference_to_alignment_columns(
                reference, rows
            )
            assert (mapping, aln_len, matched) == rma.build_reference_to_alignment_column_map(
                reference, records, self.AA_TO_CODONS, self.IGNORE
            )

    def test_ignore_alignment_chars_matches_the_original(self):
        import run_mutational_accessibility as rma_script
        assert set(common.IGNORE_ALIGNMENT_CHARS) == set(rma_script.IGNORE_ALIGNMENT_CHARS)
