#!/usr/bin/env python3
"""Tests for ``scripts/prescott_iav/run_escott.py`` -- stage C of the pipeline.

The module's job is to drive two hostile CWD-scoped command line tools and turn
their output into a matrix that ``run_mutational_accessibility.py`` can read
without any new I/O code.  The tests are therefore organised around the four
ways that silently goes wrong rather than around the public API surface:

* **naming.**  ``escott`` derives a token ``prot`` from the FIRST FASTA header by
  splitting on the first non-alphanumeric character, then writes ``<prot>.fasta``,
  ``<prot>_jet.res`` and ``<prot>_normPred_evolCombi.txt`` as *bare filenames in
  the current directory*.  ``BLAT/1-286`` becomes ``BLAT`` and ``J.2.4`` becomes
  ``J``, so a staged input that lands on one of those names is destroyed before
  it is read and the run still exits 0.

* **parsing.**  The product is R ``write.table`` output: the header carries L
  quoted names while every data row carries L+1 fields, and the wild-type cell of
  each column is the bare token ``NA``.  A parser that assumes a square header
  shifts every column by one and nothing downstream notices.

* **the transform.**  ``P = softmax(E/T)`` per column is what makes the alpha
  sweep's arithmetic come out as an honest linear trade-off.  Columns must sum to
  1, the result must be invariant to a per-column additive constant (that is the
  entire justification for the choice), and it must survive the value ranges real
  ESCOTT produces.

* **caching.**  Every stage is keyed on a content hash.  A cache that ignores a
  changed input silently serves a stale answer for the rest of the project.

Fast tests mock ``escott``/``prescott`` (see :class:`ToolRecorder`); the handful
that run the real binaries are marked ``slow`` + ``requires_escott`` and are
opt-in behind ``--run-slow``.

Run with::

    /home3/oml4h/miniconda3/envs/PRESCOTT/bin/python -m pytest \
        /home3/oml4h/PLM_SARS-CoV-2/tests_prescott_iav/test_run_escott.py -q
"""

from __future__ import annotations

import json
import re as _re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prescott_iav import constants  # noqa: E402
from prescott_iav import run_escott as R  # noqa: E402
from tests_prescott_iav import conftest as C  # noqa: E402


# --------------------------------------------------------------------------- #
# Independent re-implementations, so a test is never validated by the code it
# is testing.
# --------------------------------------------------------------------------- #

def independent_prot_token(header: str) -> str:
    """escott.py:76 rewritten from the description, not imported.

    ``re.compile("[^A-Z0-9a-z]").split(header)[0]`` -- everything up to the first
    character that is not a letter or a digit.
    """
    return _re.split(r"[^A-Za-z0-9]", header)[0]


def independent_softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    """Column-wise softmax written the naive way, for cross-checking."""
    out = np.empty_like(values, dtype=float)
    for column in range(values.shape[1]):
        scaled = [float(v) / temperature for v in values[:, column]]
        top = max(scaled)
        exponentiated = [np.exp(v - top) for v in scaled]
        total = sum(exponentiated)
        out[:, column] = [v / total for v in exponentiated]
    return out


PLM_ROWS = list(C.PLM_CACHE_ROW_ORDER)


def small_frame(values: Sequence[Sequence[float]], columns: Sequence[int] = (1, 2)) -> pd.DataFrame:
    """A tiny frame for hand-checked arithmetic.

    ``apply_prescott_equation`` is pure elementwise numpy, so it does not care
    that this is 2 x 2 rather than 20 x L -- which is exactly what makes the
    equations checkable by hand.
    """
    index = PLM_ROWS[: len(values)]
    return pd.DataFrame(np.asarray(values, dtype=float), index=index, columns=list(columns))


# --------------------------------------------------------------------------- #
# The subprocess stand-in.
# --------------------------------------------------------------------------- #

class ToolRecorder:
    """Replacement for ``run_escott.subprocess.run`` covering escott AND prescott.

    It records every invocation and reproduces the side effects that matter:

    * escott writes ``<prot>.fasta`` and copies ``--jetfile`` onto
      ``<prot>_jet.res`` **in the CWD** (escott.py:77-85, :1112-1113), which is
      the entire reason stage C stages its inputs under different names;
    * escott then writes ``<prot>_normPred_evolCombi.txt``;
    * prescott writes ``<-o>-details.csv``.

    ``rewrite_jet`` makes the fake clobber ``<prot>_jet.res`` with garbage, so a
    test can prove stage B's own file survived.
    """

    def __init__(
        self,
        *,
        protein: str = C.QUERY_PROTEIN,
        flat_positions: Sequence[int] = C.ESCOTT_FLAT_POSITIONS,
        returncode: int = 0,
        log: str = "escott finished",
        write_product: bool = True,
        product_line_limit: Optional[int] = None,
        rewrite_jet: bool = True,
        details_rows: Optional[pd.DataFrame] = None,
        prescott_returncode: int = 0,
        write_details: bool = True,
    ) -> None:
        self.protein = protein
        self.flat_positions = tuple(flat_positions)
        self.returncode = returncode
        self.log = log
        self.write_product = write_product
        self.product_line_limit = product_line_limit
        self.rewrite_jet = rewrite_jet
        self.details_rows = details_rows
        self.prescott_returncode = prescott_returncode
        self.write_details = write_details
        self.calls: List[Dict[str, object]] = []

    # -- the callable --------------------------------------------------- #
    def __call__(self, cmd, **kwargs):
        cwd = Path(kwargs["cwd"])
        self.calls.append({"cmd": list(cmd), "cwd": cwd, "env": dict(kwargs.get("env") or {})})
        if "--jetfile" in cmd:
            return self._escott(list(cmd), cwd)
        if "-e" in cmd:
            return self._prescott(list(cmd), cwd)
        raise AssertionError(f"unexpected command {cmd!r}")

    # -- helpers -------------------------------------------------------- #
    @property
    def escott_calls(self) -> List[Dict[str, object]]:
        return [call for call in self.calls if "--jetfile" in call["cmd"]]

    @property
    def prescott_calls(self) -> List[Dict[str, object]]:
        return [call for call in self.calls if "-e" in call["cmd"]]

    def _escott(self, cmd: List[str], cwd: Path):
        msa_name = cmd[1]
        header = (cwd / msa_name).read_text(encoding="utf-8").splitlines()[0][1:]
        prot = independent_prot_token(header)
        # escott.py:77-85 -- <prot>.fasta is opened for WRITING.
        (cwd / f"{prot}.fasta").write_text(f">{prot}\n{self.protein}\n", encoding="utf-8")
        if self.rewrite_jet:
            # escott.py:1112-1113 + :1187 -- the jetfile is copied onto
            # <prot>_jet.res and the structural modes rewrite it in place.
            (cwd / f"{prot}_jet.res").write_text("CLOBBERED BY ESCOTT\n", encoding="utf-8")
        if self.write_product:
            product = cwd / f"{prot}_normPred_evolCombi.txt"
            C.write_escott_normpred(product, self.protein, self.flat_positions)
            if self.product_line_limit is not None:
                lines = product.read_text(encoding="utf-8").splitlines()
                product.write_text(
                    "\n".join(lines[: self.product_line_limit]) + "\n", encoding="utf-8"
                )
        return _Completed(self.returncode, self.log, "")

    def _prescott(self, cmd: List[str], cwd: Path):
        stem = cmd[cmd.index("-o") + 1]
        if self.write_details:
            frame = self.details_rows
            if frame is None:
                frame = pd.DataFrame(
                    [{"mutant": "M1K", "ESCOTT": 0.5, "protein": "X",
                      "log10frequency": -1.0, "labels": "", "position": 1,
                      "Selected Population": "", "PRESCOTT": 0.5}]
                )
            frame.to_csv(cwd / f"{stem}-details.csv", index=False)
        return _Completed(self.prescott_returncode, "prescott done", "")


class _Completed:
    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture()
def fake_tools(monkeypatch):
    """Install a :class:`ToolRecorder` and hand it back for inspection."""
    def _install(**kwargs) -> ToolRecorder:
        recorder = ToolRecorder(**kwargs)
        monkeypatch.setattr(R.subprocess, "run", recorder)
        return recorder
    return _install


@pytest.fixture()
def escott_inputs(tmp_path: Path) -> Dict[str, Path]:
    """A minimal stage A/B product pair: one MSA and one surrogate jet table."""
    msa = C.write_fasta(
        tmp_path / "msa_K.fasta",
        [(C.QUERY_HEADER, C.QUERY_PROTEIN)]
        + list(zip(C.TINY_MSA_HEADERS[1:], C.TINY_MSA_ROWS[1:])),
    )
    jet = C.write_jet_res(tmp_path / "K_surrogate_jet.res")
    return {"msa": msa, "jet": jet, "workdir": tmp_path / "escott" / "K"}


@pytest.fixture()
def parsed_matrix(tmp_path: Path) -> pd.DataFrame:
    """The fixture ESCOTT product, parsed by the module under test."""
    path = C.write_escott_normpred(tmp_path / f"{C.QUERY_HEADER}_normPred_evolCombi.txt")
    return R.read_escott_matrix(path)


# =========================================================================== #
# 1. The escott `prot` token and the CWD naming constraint
# =========================================================================== #

@pytest.mark.unit
class TestEscottProtToken:
    """``prot`` decides every output filename, so it is derived, never guessed."""

    @pytest.mark.parametrize(
        "header, expected",
        [
            ("BLAT/1-286", "BLAT"),          # the shipped PRESCOTT alignment
            ("J.2.4", "J"),                  # a lineage label: truncated at the dot
            ("J.2_int", "J"),                # ... and at the underscore
            ("HAK", "HAK"),                  # what stage A actually writes
            ("HAJ24 A/England/415/2024", "HAJ24"),
            ("EPI4748783|HA|A/England/01837755/2025", "EPI4748783"),
            ("K", "K"),
        ],
    )
    def test_token_matches_escotts_own_split_rule(self, tmp_path, header, expected):
        path = C.write_fasta(tmp_path / "q.fasta", [(header, "MKT")])
        assert R.escott_prot_token(path) == expected
        # and the module agrees with the rule re-derived from escott.py:76
        assert R.escott_prot_token(path) == independent_prot_token(header)

    def test_slash_header_would_have_collided_with_a_directory(self, tmp_path):
        """``BLAT/1-286`` -> ``BLAT``: the '/' must NOT survive into a filename."""
        path = C.write_fasta(tmp_path / "q.fasta", [("BLAT/1-286", "MKT")])
        token = R.escott_prot_token(path)
        assert "/" not in token
        assert (tmp_path / f"{token}_normPred_evolCombi.txt").name == "BLAT_normPred_evolCombi.txt"

    def test_dotted_header_would_have_broken_splitext(self, tmp_path):
        """``J.2.4`` -> ``J``: no dot survives, so os.path.splitext cannot truncate."""
        path = C.write_fasta(tmp_path / "q.fasta", [("J.2.4", "MKT")])
        assert "." not in R.escott_prot_token(path)

    def test_missing_gt_is_rejected_with_escotts_own_error_name(self, tmp_path):
        path = tmp_path / "bad.fasta"
        path.write_text("MKTIIALSYI\n", encoding="utf-8")
        with pytest.raises(ValueError, match="bad FASTA format"):
            R.escott_prot_token(path)

    def test_header_starting_with_punctuation_yields_an_empty_token(self, tmp_path):
        """``>|EPI123`` splits to '' -- escott would write ``_normPred...``."""
        path = C.write_fasta(tmp_path / "q.fasta", [("|EPI123", "MKT")])
        with pytest.raises(ValueError, match="empty escott token"):
            R.escott_prot_token(path)

    def test_prepared_tree_tokens_are_distinct_per_lineage(self, prepared_inputs_tree):
        """Three lineages start with 'J'; their MSA headers must still differ.

        If stage A ever wrote the raw label as the header, ``J_int``, ``J.2_int``
        and ``J.2.4`` would ALL yield ``prot == 'J'``.
        """
        inputs = Path(prepared_inputs_tree["inputs_dir"])
        tokens = {
            label: R.escott_prot_token(inputs / "msa" / f"msa_{label}.fasta")
            for label in prepared_inputs_tree["lineages"]
        }
        assert tokens == prepared_inputs_tree["prot_tokens"]
        assert len(set(tokens.values())) == len(tokens), tokens


@pytest.mark.unit
class TestInputStaging:
    """No staged input may land on a filename escott owns."""

    def test_staged_names_are_returned_and_content_copied(self, tmp_path, escott_inputs):
        workdir = R.ensure_dir(tmp_path / "wd")
        msa, jet = R._stage_escott_inputs(
            workdir, escott_inputs["msa"], escott_inputs["jet"], "HAK"
        )
        assert msa == workdir / "ha_msa_msa_K.fasta"
        assert jet == workdir / "K_surrogate_jet.res"
        assert msa.read_bytes() == escott_inputs["msa"].read_bytes()
        assert jet.read_bytes() == escott_inputs["jet"].read_bytes()

    @pytest.mark.parametrize("forbidden", ["HAK_jet.res", "HAK.fasta", "HAK.pdb"])
    def test_a_jet_file_named_like_an_escott_output_is_refused(
        self, tmp_path, escott_inputs, forbidden
    ):
        """These three names are destroyed by escott before they are read."""
        collider = tmp_path / forbidden
        shutil.copy2(escott_inputs["jet"], collider)
        with pytest.raises(ValueError, match="collides with an escott-owned filename"):
            R._stage_escott_inputs(
                R.ensure_dir(tmp_path / "wd"), escott_inputs["msa"], collider, "HAK"
            )

    def test_the_same_name_is_fine_for_a_different_prot(self, tmp_path, escott_inputs):
        """The guard is prot-relative, not a blanket ban on '_jet.res'."""
        collider = tmp_path / "HAK_jet.res"
        shutil.copy2(escott_inputs["jet"], collider)
        msa, jet = R._stage_escott_inputs(
            R.ensure_dir(tmp_path / "wd"), escott_inputs["msa"], collider, "HAJ24"
        )
        assert jet.name == "HAK_jet.res"

    def test_workdir_is_one_directory_per_lineage(self, tmp_path):
        root = tmp_path / "escott"
        first = R.escott_workdir(root, "J.2_int")
        second = R.escott_workdir(root, "J.2.4")
        assert first != second
        assert first.is_dir() and second.is_dir()
        assert first.parent == second.parent == root


# =========================================================================== #
# 2. Small helpers
# =========================================================================== #

@pytest.mark.unit
class TestSmallHelpers:

    def test_safe_label_keeps_dots_and_kills_slashes(self):
        assert R.safe_label("J.2_int") == "J.2_int"
        assert R.safe_label(" A/B ") == "A-B"

    def test_dotfree_key_protects_prescotts_splitext(self):
        assert R.dotfree_key("J.2_int") == "J_2_int"
        assert R.dotfree_key("J.2.4") == "J_2_4"
        assert R.dotfree_key("K") == "K"

    def test_file_md5_matches_an_independent_digest(self, tmp_path):
        path = tmp_path / "x.bin"
        path.write_bytes(b"abcdef" * 1000)
        assert R.file_md5(path) == C.md5_file(path)

    def test_content_hash_is_stable_and_content_sensitive(self, tmp_path):
        a = tmp_path / "a.txt"
        a.write_text("one", encoding="utf-8")
        first = R.content_hash([a])
        assert R.content_hash([a]) == first
        a.write_text("two", encoding="utf-8")
        assert R.content_hash([a]) != first

    def test_content_hash_ignores_the_directory_but_not_the_basename(self, tmp_path):
        (tmp_path / "d1").mkdir()
        (tmp_path / "d2").mkdir()
        (tmp_path / "d1" / "same.txt").write_text("payload", encoding="utf-8")
        (tmp_path / "d2" / "same.txt").write_text("payload", encoding="utf-8")
        (tmp_path / "d2" / "other.txt").write_text("payload", encoding="utf-8")
        assert R.content_hash([tmp_path / "d1" / "same.txt"]) == R.content_hash(
            [tmp_path / "d2" / "same.txt"]
        )
        assert R.content_hash([tmp_path / "d1" / "same.txt"]) != R.content_hash(
            [tmp_path / "d2" / "other.txt"]
        )

    def test_content_hash_covers_the_scalar_arguments(self, tmp_path):
        path = tmp_path / "a.txt"
        path.write_text("payload", encoding="utf-8")
        assert R.content_hash([path], {"alphabet": "lw-i.7"}) != R.content_hash(
            [path], {"alphabet": "lw-i.6"}
        )
        # key order must not matter -- json.dumps(sort_keys=True)
        assert R.content_hash([path], {"a": 1, "b": 2}) == R.content_hash([path], {"b": 2, "a": 1})

    def test_content_hash_is_order_sensitive_across_files(self, tmp_path):
        first = tmp_path / "one.txt"
        second = tmp_path / "two.txt"
        first.write_text("1", encoding="utf-8")
        second.write_text("2", encoding="utf-8")
        assert R.content_hash([first, second]) != R.content_hash([second, first])

    def test_ensure_dir_is_idempotent(self, tmp_path):
        target = tmp_path / "a" / "b"
        assert R.ensure_dir(target) == target
        assert R.ensure_dir(target).is_dir()

    def test_parse_grid_casts_and_drops_blanks(self):
        assert R.parse_grid("0.25, 0.5,1.0", float) == [0.25, 0.5, 1.0]
        assert R.parse_grid("2,,3", int) == [2, 3]
        assert R.parse_grid("", int) == []


@pytest.mark.unit
class TestReadSingleFasta:

    def test_returns_header_and_ungapped_uppercase_sequence(self, tmp_path):
        path = tmp_path / "q.fasta"
        path.write_text(">HAK descr\nmkt-ii\nalsy.i\n", encoding="utf-8")
        header, sequence = R.read_single_fasta(path)
        assert header == "HAK descr"
        assert sequence == "MKTIIALSYI"

    def test_two_records_are_refused(self, tmp_path):
        path = C.write_fasta(tmp_path / "q.fasta", [("a", "MKT"), ("b", "MKT")])
        with pytest.raises(ValueError, match="more than one record"):
            R.read_single_fasta(path)

    def test_no_record_is_refused(self, tmp_path):
        path = tmp_path / "q.fasta"
        path.write_text("\n\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no FASTA record"):
            R.read_single_fasta(path)


@pytest.mark.unit
class TestPrescottEnv:
    """escott shells out to Rscript/mkdssp by bare name; PATH is not optional."""

    def test_bin_dir_is_prepended_and_headless_backend_forced(self):
        env = R.prescott_env(Path("/nonexistent/prescott/bin"))
        assert env["PATH"].split(":")[0] == "/nonexistent/prescott/bin"
        assert env["MPLBACKEND"] == "Agg"
        assert env["R_LIBS_USER"] == ""

    def test_default_is_the_prescott_env_bin(self):
        env = R.prescott_env()
        assert str(R.DEFAULT_PRESCOTT_ENV_BIN) in env["PATH"].split(":")

    def test_an_already_present_bin_dir_is_not_duplicated(self, monkeypatch):
        monkeypatch.setenv("PATH", "/x/bin:/y/bin")
        env = R.prescott_env(Path("/x/bin"))
        assert env["PATH"].split(":").count("/x/bin") == 1
        assert env["PATH"] == "/x/bin:/y/bin"


# =========================================================================== #
# 3. run_escott_for_lineage: command construction, caching, failure detection
# =========================================================================== #

@pytest.mark.unit
class TestAssertEscottSucceeded:
    """escott exits 0 on R failure (escott.py:360 discards the return code)."""

    def test_returns_the_product_when_it_exists_and_the_log_is_clean(self, tmp_path):
        product = C.write_escott_normpred(tmp_path / "HAK_normPred_evolCombi.txt")
        assert R._assert_escott_succeeded(tmp_path, "HAK", "all good") == product

    @pytest.mark.parametrize("signature", list(R.ESCOTT_FAILURE_SIGNATURES))
    def test_every_known_r_failure_signature_is_caught(self, tmp_path, signature):
        # The product is present, so only the log inspection can catch this.
        C.write_escott_normpred(tmp_path / "HAK_normPred_evolCombi.txt")
        with pytest.raises(RuntimeError, match="R-side failure"):
            R._assert_escott_succeeded(tmp_path, "HAK", f"blah\n{signature} something\nblah")

    def test_missing_product_is_reported_with_the_log_tail(self, tmp_path):
        log = "\n".join(f"line {i}" for i in range(60))
        with pytest.raises(RuntimeError) as excinfo:
            R._assert_escott_succeeded(tmp_path, "HAK", log)
        message = str(excinfo.value)
        assert "produced no HAK_normPred_evolCombi.txt" in message
        assert "line 59" in message and "line 34" not in message  # last 25 lines only


@pytest.mark.unit
class TestRunEscottForLineage:

    def test_command_cwd_and_environment(self, escott_inputs, fake_tools):
        recorder = fake_tools()
        product = R.run_escott_for_lineage(
            "K", escott_inputs["msa"], escott_inputs["jet"], escott_inputs["workdir"]
        )
        assert product.name == "HAK_normPred_evolCombi.txt"
        assert len(recorder.escott_calls) == 1
        call = recorder.escott_calls[0]
        assert call["cmd"] == [
            str(R.DEFAULT_ESCOTT_BIN),
            "ha_msa_msa_K.fasta",
            "--jetfile", "K_surrogate_jet.res",
            "--alphabet", "lw-i.7",
            "--maxcoillength", "5",
        ]
        # Bare filenames, not paths: escott is a CWD tool and the CWD is the workdir.
        assert call["cwd"] == escott_inputs["workdir"]
        assert call["env"]["MPLBACKEND"] == "Agg"

    def test_no_pdbfile_means_no_pdb_and_no_cvrc_flag(self, escott_inputs, fake_tools):
        recorder = fake_tools()
        R.run_escott_for_lineage(
            "K", escott_inputs["msa"], escott_inputs["jet"], escott_inputs["workdir"]
        )
        assert "--pdbfile" not in recorder.escott_calls[0]["cmd"]
        assert "--cvrc" not in recorder.escott_calls[0]["cmd"]

    def test_alphabet_and_maxcoillength_are_forwarded(self, escott_inputs, fake_tools):
        recorder = fake_tools()
        R.run_escott_for_lineage(
            "K", escott_inputs["msa"], escott_inputs["jet"], escott_inputs["workdir"],
            alphabet="lw-i.6", max_coil_length=9,
        )
        cmd = recorder.escott_calls[0]["cmd"]
        assert cmd[cmd.index("--alphabet") + 1] == "lw-i.6"
        assert cmd[cmd.index("--maxcoillength") + 1] == "9"

    def test_cv_radius_becomes_cvrc(self, escott_inputs, fake_tools):
        recorder = fake_tools()
        R.run_escott_for_lineage(
            "K", escott_inputs["msa"], escott_inputs["jet"], escott_inputs["workdir"],
            cv_radius=7.5,
        )
        cmd = recorder.escott_calls[0]["cmd"]
        assert cmd[cmd.index("--cvrc") + 1] == "7.5"

    def test_pdbfile_is_absolute_and_carries_the_loud_warning(
        self, tmp_path, escott_inputs, fake_tools, capsys
    ):
        pdb = tmp_path / "6WXB_chainA_qnum.pdb"
        pdb.write_text(C.build_pdb(C.cv_ladder_atoms()), encoding="utf-8")
        recorder = fake_tools()
        R.run_escott_for_lineage(
            "K", escott_inputs["msa"], escott_inputs["jet"], escott_inputs["workdir"],
            pdbfile=pdb,
        )
        cmd = recorder.escott_calls[0]["cmd"]
        assert cmd[cmd.index("--pdbfile") + 1] == str(pdb.resolve())
        out = capsys.readouterr().out
        assert "sstjetormaxtwocomponent" in out
        assert "RECOMPUTE" in out

    def test_marker_records_the_effective_normweightmode(self, escott_inputs, fake_tools):
        """escott.main() overwrites --normweightmode unconditionally, so the marker
        records what escott WILL do, not what the caller asked for."""
        fake_tools()
        R.run_escott_for_lineage(
            "K", escott_inputs["msa"], escott_inputs["jet"], escott_inputs["workdir"]
        )
        marker = json.loads((escott_inputs["workdir"] / "escott_exit.json").read_text())
        assert marker["escott_args"]["effective_normweightmode"] == "tjet"
        assert marker["escott_args"]["pdbfile"] is None
        assert marker["prot"] == "HAK"
        assert marker["returncode"] == 0
        assert marker["msa_path"] == str(escott_inputs["msa"])
        assert marker["jet_path"] == str(escott_inputs["jet"])
        assert "normweightmode" not in marker["escott_args"]

    def test_pdbfile_changes_the_effective_normweightmode(
        self, tmp_path, escott_inputs, fake_tools
    ):
        pdb = tmp_path / "s.pdb"
        pdb.write_text(C.build_pdb(C.cv_ladder_atoms()), encoding="utf-8")
        fake_tools()
        R.run_escott_for_lineage(
            "K", escott_inputs["msa"], escott_inputs["jet"], escott_inputs["workdir"],
            pdbfile=pdb,
        )
        marker = json.loads((escott_inputs["workdir"] / "escott_exit.json").read_text())
        assert marker["escott_args"]["effective_normweightmode"] == "sstjetormaxtwocomponent"
        assert marker["escott_args"]["pdbfile"] == "s.pdb"

    def test_log_is_captured_to_disk(self, escott_inputs, fake_tools):
        fake_tools(log="hello from escott")
        R.run_escott_for_lineage(
            "K", escott_inputs["msa"], escott_inputs["jet"], escott_inputs["workdir"]
        )
        assert "hello from escott" in (escott_inputs["workdir"] / "escott.log").read_text()

    def test_stage_b_jet_table_is_never_mutated(self, escott_inputs, fake_tools):
        """escott rewrites ``<prot>_jet.res``; our copy must be a different file."""
        before = C.md5_file(escott_inputs["jet"])
        fake_tools(rewrite_jet=True)
        R.run_escott_for_lineage(
            "K", escott_inputs["msa"], escott_inputs["jet"], escott_inputs["workdir"]
        )
        assert C.md5_file(escott_inputs["jet"]) == before
        clobbered = escott_inputs["workdir"] / "HAK_jet.res"
        assert clobbered.read_text().strip() == "CLOBBERED BY ESCOTT"
        assert (escott_inputs["workdir"] / "K_surrogate_jet.res").read_bytes() == \
            escott_inputs["jet"].read_bytes()

    def test_r_failure_with_exit_zero_is_still_a_failure(self, escott_inputs, fake_tools):
        fake_tools(write_product=False, log="Error in eval: non-conformable arguments")
        with pytest.raises(RuntimeError, match="R-side failure"):
            R.run_escott_for_lineage(
                "K", escott_inputs["msa"], escott_inputs["jet"], escott_inputs["workdir"]
            )
        # No marker, so the next run recomputes rather than serving the failure.
        assert not (escott_inputs["workdir"] / "escott_exit.json").exists()


@pytest.mark.unit
class TestEscottCaching:
    """Content-hash caching: same inputs -> no recompute, changed inputs -> recompute."""

    def _run(self, escott_inputs, **kwargs):
        return R.run_escott_for_lineage(
            "K", escott_inputs["msa"], escott_inputs["jet"], escott_inputs["workdir"], **kwargs
        )

    def test_identical_inputs_do_not_respawn_escott(self, escott_inputs, fake_tools, capsys):
        recorder = fake_tools()
        first = self._run(escott_inputs)
        capsys.readouterr()
        second = self._run(escott_inputs)
        assert first == second
        assert len(recorder.escott_calls) == 1
        assert "cache hit" in capsys.readouterr().out

    def test_touching_the_input_is_free(self, escott_inputs, fake_tools):
        recorder = fake_tools()
        self._run(escott_inputs)
        escott_inputs["jet"].touch()
        self._run(escott_inputs)
        assert len(recorder.escott_calls) == 1

    def test_changed_jet_content_forces_a_recompute(self, escott_inputs, fake_tools):
        recorder = fake_tools()
        self._run(escott_inputs)
        C.write_jet_res(escott_inputs["jet"], zero_trace_positions=(1, 2, 3))
        self._run(escott_inputs)
        assert len(recorder.escott_calls) == 2

    def test_changed_msa_content_forces_a_recompute(self, escott_inputs, fake_tools):
        recorder = fake_tools()
        self._run(escott_inputs)
        C.write_fasta(
            escott_inputs["msa"],
            [(C.QUERY_HEADER, C.QUERY_PROTEIN)]
            + list(zip(C.TINY_MSA_HEADERS[1:5], C.TINY_MSA_ROWS[1:5])),
        )
        self._run(escott_inputs)
        assert len(recorder.escott_calls) == 2

    def test_changed_escott_argument_forces_a_recompute(self, escott_inputs, fake_tools):
        recorder = fake_tools()
        self._run(escott_inputs)
        self._run(escott_inputs, alphabet="lw-i.6")
        assert len(recorder.escott_calls) == 2

    def test_adding_a_pdbfile_forces_a_recompute(self, tmp_path, escott_inputs, fake_tools):
        recorder = fake_tools()
        self._run(escott_inputs)
        pdb = tmp_path / "s.pdb"
        pdb.write_text(C.build_pdb(C.cv_ladder_atoms()), encoding="utf-8")
        self._run(escott_inputs, pdbfile=pdb)
        assert len(recorder.escott_calls) == 2
        # ... and changing the structure's CONTENT under the same name too
        pdb.write_text(C.build_pdb(C.cv_ladder_atoms("B")), encoding="utf-8")
        self._run(escott_inputs, pdbfile=pdb)
        assert len(recorder.escott_calls) == 3

    def test_force_ignores_a_valid_cache(self, escott_inputs, fake_tools):
        recorder = fake_tools()
        self._run(escott_inputs)
        self._run(escott_inputs, force=True)
        assert len(recorder.escott_calls) == 2

    def test_a_truncated_product_is_not_served_from_cache(self, escott_inputs, fake_tools):
        """21 lines == 1 header + 20 amino-acid rows; anything else is a torn write."""
        recorder = fake_tools()
        self._run(escott_inputs)
        product = escott_inputs["workdir"] / "HAK_normPred_evolCombi.txt"
        lines = product.read_text().splitlines()
        product.write_text("\n".join(lines[:15]) + "\n", encoding="utf-8")
        self._run(escott_inputs)
        assert len(recorder.escott_calls) == 2
        assert len(product.read_text().splitlines()) == 21

    def test_a_corrupt_marker_is_not_served_from_cache(self, escott_inputs, fake_tools):
        recorder = fake_tools()
        self._run(escott_inputs)
        (escott_inputs["workdir"] / "escott_exit.json").write_text("{not json", encoding="utf-8")
        self._run(escott_inputs)
        assert len(recorder.escott_calls) == 2

    def test_a_missing_marker_is_not_served_from_cache(self, escott_inputs, fake_tools):
        recorder = fake_tools()
        self._run(escott_inputs)
        (escott_inputs["workdir"] / "escott_exit.json").unlink()
        self._run(escott_inputs)
        assert len(recorder.escott_calls) == 2


# =========================================================================== #
# 4. Parsing the normPred matrix
# =========================================================================== #

@pytest.mark.unit
class TestReadEscottMatrix:
    """R ``write.table`` output: L header fields, L+1 row fields, bare NA on the WT."""

    def test_the_fixture_file_really_is_in_the_asymmetric_r_format(self, fake_escott_matrix):
        lines = Path(fake_escott_matrix["path"]).read_text().splitlines()
        assert len(lines) == 21
        header_fields = lines[0].split()
        row_fields = lines[1].split()
        assert len(header_fields) == C.QUERY_LENGTH
        assert len(row_fields) == C.QUERY_LENGTH + 1      # row name + L values
        assert header_fields[0] == '"M1"' and row_fields[0] == '"a"'
        assert "NA" in row_fields                          # bare, unquoted

    def test_round_trip_of_every_cell(self, fake_escott_matrix):
        frame = R.read_escott_matrix(fake_escott_matrix["path"])
        assert frame.shape == (20, C.QUERY_LENGTH)
        assert list(frame.index) == list(C.PLM_CACHE_ROW_ORDER)
        assert list(frame.columns) == list(range(1, C.QUERY_LENGTH + 1))
        expected = fake_escott_matrix["values"]
        for (aa, position), value in expected.items():
            got = frame.at[aa, position]
            if value is None:
                assert np.isnan(got), (aa, position)
            else:
                assert got == pytest.approx(value, abs=0.0), (aa, position)

    def test_rows_are_reindexed_out_of_escott_order_into_plm_order(self, fake_escott_matrix):
        frame = R.read_escott_matrix(fake_escott_matrix["path"])
        assert list(frame.index) != [aa.upper() for aa in C.ESCOTT_ROW_ORDER]
        assert list(frame.index) == list(C.PLM_CACHE_ROW_ORDER)
        assert sorted(frame.index) == sorted(aa.upper() for aa in C.ESCOTT_ROW_ORDER)

    def test_wild_type_sequence_is_recovered_from_the_column_labels(self, fake_escott_matrix):
        frame = R.read_escott_matrix(fake_escott_matrix["path"])
        assert frame.attrs["wt_sequence"] == C.QUERY_PROTEIN
        assert R.escott_wt_sequence(frame) == C.QUERY_PROTEIN
        assert frame.attrs["source_path"] == str(fake_escott_matrix["path"])

    def test_exactly_one_nan_per_column_and_it_sits_on_the_wild_type(self, fake_escott_matrix):
        frame = R.read_escott_matrix(fake_escott_matrix["path"])
        assert set(frame.isna().sum(axis=0).unique()) == {1}
        for position, wt in enumerate(C.QUERY_PROTEIN, start=1):
            assert np.isnan(frame.at[wt, position])

    def test_expect_protein_accepts_the_matching_reference(self, fake_escott_matrix):
        frame = R.read_escott_matrix(fake_escott_matrix["path"], expect_protein=C.QUERY_PROTEIN)
        assert frame.attrs["wt_sequence"] == C.QUERY_PROTEIN

    def test_expect_protein_rejects_a_one_residue_disagreement(self, escott_matrix_factory):
        """This is the guard against scoring lineage A's matrix as lineage B's."""
        path, _ = escott_matrix_factory()
        with pytest.raises(ValueError) as excinfo:
            R.read_escott_matrix(path, expect_protein=C.PARENT_PROTEIN)
        assert "40:T!=I" in str(excinfo.value)

    def test_a_square_header_is_refused_rather_than_misread(self, tmp_path):
        """The asymmetry is load bearing: a leading header field shifts every column."""
        protein = "MKT"
        lines = ['"aa" ' + " ".join(f'"{protein[i]}{i + 1}"' for i in range(3))]
        for aa in C.ESCOTT_ROW_ORDER:
            cells = ["NA" if aa.upper() == protein[i] else "-1.0" for i in range(3)]
            lines.append(f'"{aa}" ' + " ".join(cells))
        path = tmp_path / "sq_normPred_evolCombi.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="unexpected row labels"):
            R.read_escott_matrix(path)

    def test_wrong_row_count_is_refused(self, tmp_path):
        path = C.write_escott_normpred(tmp_path / "m_normPred_evolCombi.txt", "MKT", ())
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines[:-2]) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="expected 20 amino-acid rows"):
            R.read_escott_matrix(path)

    def test_unlabelled_v1_vn_columns_are_refused(self, tmp_path):
        """Real ``write.table`` output without dimnames looks like ``"V1" "V2" ...``.

        ``V1`` matches the ``<WT><pos>`` regex, so only the NA-placement check can
        catch it -- and it must, because reading it would silently invent a
        wild-type sequence of 'VVVV...'.
        """
        protein = "MKT"
        lines = [" ".join(f'"V{i + 1}"' for i in range(3))]
        for aa in C.ESCOTT_ROW_ORDER:
            cells = ["NA" if aa.upper() == protein[i] else "-1.0" for i in range(3)]
            lines.append(f'"{aa}" ' + " ".join(cells))
        path = tmp_path / "v_normPred_evolCombi.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="the NA in column V1 is not on row V"):
            R.read_escott_matrix(path)

    def test_two_nas_in_one_column_are_refused(self, escott_matrix_factory):
        values = dict(C.escott_matrix_values("MKT", ()))
        values[("A", 2)] = None
        path, _ = escott_matrix_factory(protein="MKT", values=values)
        with pytest.raises(ValueError, match="expected exactly one NA per column"):
            R.read_escott_matrix(path)

    def test_zero_nas_in_one_column_are_refused(self, escott_matrix_factory):
        values = dict(C.escott_matrix_values("MKT", ()))
        values[("M", 1)] = -1.0
        path, _ = escott_matrix_factory(protein="MKT", values=values)
        with pytest.raises(ValueError, match="expected exactly one NA per column"):
            R.read_escott_matrix(path)

    def test_an_na_off_the_wild_type_row_is_refused(self, escott_matrix_factory):
        values = dict(C.escott_matrix_values("MKT", ()))
        values[("M", 1)] = -1.0
        values[("A", 1)] = None
        path, _ = escott_matrix_factory(protein="MKT", values=values)
        with pytest.raises(ValueError, match="the NA in column M1 is not on row M"):
            R.read_escott_matrix(path)

    def test_a_malformed_column_label_is_refused(self, tmp_path):
        path = C.write_escott_normpred(tmp_path / "m_normPred_evolCombi.txt", "MKT", ())
        lines = path.read_text().splitlines()
        lines[0] = lines[0].replace('"K2"', '"K2x"')
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="is not '<WT><pos>'"):
            R.read_escott_matrix(path)

    def test_non_contiguous_positions_are_refused(self, tmp_path):
        path = C.write_escott_normpred(tmp_path / "m_normPred_evolCombi.txt", "MKT", ())
        lines = path.read_text().splitlines()
        lines[0] = lines[0].replace('"T3"', '"T9"')
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="not contiguous"):
            R.read_escott_matrix(path)

    def test_escott_wt_sequence_needs_the_attr(self):
        with pytest.raises(ValueError, match="wt_sequence"):
            R.escott_wt_sequence(pd.DataFrame(np.zeros((2, 2))))


# =========================================================================== #
# 5. The wild-type fill, the softmax and the flat-column count
# =========================================================================== #

@pytest.mark.unit
class TestFillWildtype:

    def test_column_max_gives_the_wild_type_its_own_column_best(self, parsed_matrix):
        filled = R.fill_wildtype(parsed_matrix, mode="column_max")
        assert not filled.isna().any().any()
        for position, wt in enumerate(C.QUERY_PROTEIN, start=1):
            others = parsed_matrix[position].drop(index=wt)
            assert filled.at[wt, position] == pytest.approx(others.max())

    def test_global_max_reproduces_prescotts_own_choice(self, parsed_matrix):
        filled = R.fill_wildtype(parsed_matrix, mode="global_max")
        global_max = np.nanmax(parsed_matrix.to_numpy(dtype=float))
        for position, wt in enumerate(C.QUERY_PROTEIN, start=1):
            assert filled.at[wt, position] == pytest.approx(global_max)

    def test_only_the_nan_cells_move(self, parsed_matrix):
        filled = R.fill_wildtype(parsed_matrix)
        mask = ~parsed_matrix.isna().to_numpy()
        assert np.array_equal(
            filled.to_numpy()[mask], parsed_matrix.to_numpy()[mask]
        )

    def test_attrs_survive_the_fill(self, parsed_matrix):
        assert R.fill_wildtype(parsed_matrix).attrs["wt_sequence"] == C.QUERY_PROTEIN

    def test_unknown_mode_is_refused(self, parsed_matrix):
        with pytest.raises(ValueError, match="unknown wildtype fill mode"):
            R.fill_wildtype(parsed_matrix, mode="median")

    def test_the_input_frame_is_not_mutated(self, parsed_matrix):
        before = parsed_matrix.isna().sum().sum()
        R.fill_wildtype(parsed_matrix)
        assert parsed_matrix.isna().sum().sum() == before


@pytest.mark.unit
class TestEscottToProbability:

    def test_every_column_sums_to_one(self, parsed_matrix):
        probabilities = R.escott_to_probability(parsed_matrix)
        sums = probabilities.to_numpy().sum(axis=0)
        assert np.allclose(sums, 1.0, atol=1e-12)
        assert probabilities.shape == parsed_matrix.shape

    def test_every_probability_is_strictly_positive(self, parsed_matrix):
        """The alpha sweep feeds this straight into np.log."""
        probabilities = R.escott_to_probability(parsed_matrix)
        assert probabilities.to_numpy().min() > 0.0

    def test_it_agrees_with_a_naive_softmax(self, parsed_matrix):
        filled = R.fill_wildtype(parsed_matrix).to_numpy(dtype=float)
        expected = independent_softmax(filled, 1.0)
        got = R.escott_to_probability(parsed_matrix).to_numpy()
        assert np.allclose(got, expected, atol=1e-15)

    def test_invariance_to_a_per_column_additive_constant(self, parsed_matrix):
        """This is the whole justification for the transform: the sweep's score is
        ``E/T + alpha*log(mut_prob) + c_i`` with ``c_i`` constant per site."""
        rng = np.random.default_rng(20260805)
        shift = rng.normal(scale=7.0, size=parsed_matrix.shape[1])
        shifted = parsed_matrix.add(pd.Series(shift, index=parsed_matrix.columns), axis=1)
        shifted.attrs.update(parsed_matrix.attrs)
        assert np.allclose(
            R.escott_to_probability(shifted).to_numpy(),
            R.escott_to_probability(parsed_matrix).to_numpy(),
            atol=1e-12,
        )

    def test_a_global_additive_constant_is_also_absorbed(self, parsed_matrix):
        shifted = parsed_matrix + 1234.5
        shifted.attrs.update(parsed_matrix.attrs)
        assert np.allclose(
            R.escott_to_probability(shifted).to_numpy(),
            R.escott_to_probability(parsed_matrix).to_numpy(),
            atol=1e-12,
        )

    def test_temperature_scales_the_log_odds_exactly(self, parsed_matrix):
        """log(p_a/p_b) == (E_a - E_b)/T, which is the definition."""
        filled = R.fill_wildtype(parsed_matrix)
        for temperature in (0.5, 1.0, 2.0):
            probabilities = R.escott_to_probability(parsed_matrix, temperature=temperature)
            position = 1
            a, b = "A", "C"
            expected = (filled.at[a, position] - filled.at[b, position]) / temperature
            got = np.log(probabilities.at[a, position] / probabilities.at[b, position])
            assert got == pytest.approx(expected, abs=1e-9)

    def test_zero_trace_columns_softmax_to_exactly_one_twentieth(
        self, parsed_matrix, fake_escott_matrix
    ):
        """pred.R:487 multiplies by trace[i], so trace == 0 is an all-zero column,
        and after the fill every entry is equal -- pure noise at that site."""
        for temperature in (0.25, 1.0, 4.0):
            probabilities = R.escott_to_probability(parsed_matrix, temperature=temperature)
            for position in fake_escott_matrix["flat_positions"]:
                column = probabilities[position].to_numpy()
                assert np.allclose(column, fake_escott_matrix["expected_flat_probability"],
                                   atol=1e-15)

    def test_non_flat_columns_are_not_uniform(self, parsed_matrix, fake_escott_matrix):
        probabilities = R.escott_to_probability(parsed_matrix)
        informative = [
            position for position in probabilities.columns
            if position not in set(fake_escott_matrix["flat_positions"])
        ]
        assert informative
        for position in informative:
            assert probabilities[position].std() > 1e-6

    def test_large_positive_values_do_not_overflow(self, escott_matrix_factory):
        """Column-max subtraction means exp() never sees an argument above 0."""
        protein = "MKT"
        values = {}
        for position, wt in enumerate(protein, start=1):
            for index, aa in enumerate(C.ESCOTT_ROW_ORDER):
                upper = aa.upper()
                values[(upper, position)] = None if upper == wt else (700.0 + index)
        path, _ = escott_matrix_factory(protein=protein, values=values)
        probabilities = R.escott_to_probability(R.read_escott_matrix(path))
        assert np.isfinite(probabilities.to_numpy()).all()
        assert np.allclose(probabilities.to_numpy().sum(axis=0), 1.0, atol=1e-12)

    def test_a_six_log_unit_spread_survives_a_realistic_temperature(self, escott_matrix_factory):
        """The shipped MLH1 product spans 6.03 log units in its widest column."""
        protein = "MKT"
        values = {}
        for position, wt in enumerate(protein, start=1):
            for index, aa in enumerate(C.ESCOTT_ROW_ORDER):
                upper = aa.upper()
                values[(upper, position)] = None if upper == wt else -6.03 * index / 19.0
        path, _ = escott_matrix_factory(protein=protein, values=values)
        matrix = R.read_escott_matrix(path)
        for temperature in (0.05, 0.1, 1.0, 5.0):
            probabilities = R.escott_to_probability(matrix, temperature=temperature)
            assert probabilities.to_numpy().min() > 0.0

    def test_underflow_is_refused_loudly_rather_than_emitting_a_zero(
        self, escott_matrix_factory
    ):
        """A spread beyond ~745 nats underflows exp() to exactly 0.

        The module REFUSES rather than handing a zero to ``np.log`` downstream.
        Pinned because it is the failure mode of an over-small ``--escott-temperature``
        (a 6-log-unit column needs T < 0.008 to get here).
        """
        protein = "MKT"
        values = {}
        for position, wt in enumerate(protein, start=1):
            for index, aa in enumerate(C.ESCOTT_ROW_ORDER):
                upper = aa.upper()
                values[(upper, position)] = None if upper == wt else (-800.0 if index % 2 else -1e-3)
        path, _ = escott_matrix_factory(protein=protein, values=values)
        matrix = R.read_escott_matrix(path)
        with pytest.raises(AssertionError, match="non-positive probability"):
            R.escott_to_probability(matrix)

    @pytest.mark.parametrize("temperature", [0.0, -1.0, -0.001])
    def test_non_positive_temperature_is_refused(self, parsed_matrix, temperature):
        with pytest.raises(ValueError, match="temperature must be positive"):
            R.escott_to_probability(parsed_matrix, temperature=temperature)

    def test_orientation_is_the_raw_escott_one(self, parsed_matrix):
        """High P must mean TOLERATED, so the correlation with diversity is positive."""
        position = 1
        column = parsed_matrix[position]
        best = column.idxmax()
        worst = column.idxmin()
        probabilities = R.escott_to_probability(parsed_matrix)
        assert probabilities.at[best, position] > probabilities.at[worst, position]

    def test_attrs_and_labels_survive(self, parsed_matrix):
        probabilities = R.escott_to_probability(parsed_matrix)
        assert probabilities.attrs["wt_sequence"] == C.QUERY_PROTEIN
        assert list(probabilities.index) == list(C.PLM_CACHE_ROW_ORDER)
        assert list(probabilities.columns) == list(parsed_matrix.columns)


@pytest.mark.unit
class TestCountFlatColumns:

    def test_counts_the_planted_zero_trace_positions(self, parsed_matrix, fake_escott_matrix):
        assert R.count_flat_columns(parsed_matrix) == len(fake_escott_matrix["flat_positions"])

    def test_a_matrix_with_no_flat_column_counts_zero(self, escott_matrix_factory):
        path, _ = escott_matrix_factory(flat_positions=())
        assert R.count_flat_columns(R.read_escott_matrix(path)) == 0

    def test_tolerance_is_honoured(self, escott_matrix_factory):
        protein = "MKT"
        values = dict(C.escott_matrix_values(protein, ()))
        for aa in C.PLM_CACHE_ROW_ORDER:
            if values[(aa, 2)] is not None:
                values[(aa, 2)] = 1e-13
        path, _ = escott_matrix_factory(protein=protein, values=values)
        matrix = R.read_escott_matrix(path)
        assert R.count_flat_columns(matrix) == 1              # default tol 1e-12
        assert R.count_flat_columns(matrix, tolerance=1e-14) == 0


# =========================================================================== #
# 6. Frequency files and the PRESCOTT equations
# =========================================================================== #

@pytest.mark.unit
class TestLoadFrequencyFile:

    def test_reads_the_two_column_whitespace_format(self, frequency_file_factory):
        path = frequency_file_factory()
        assert R.load_frequency_file(path) == pytest.approx(
            dict(C.EXPECTED_FREQUENCY_FILE_MIN_COUNT_1)
        )

    def test_mutants_are_upper_cased(self, tmp_path):
        path = tmp_path / "f.txt"
        path.write_text("i10k 0.1\nf15d 0.25\n", encoding="utf-8")
        assert set(R.load_frequency_file(path)) == {"I10K", "F15D"}

    def test_duplicate_keys_are_refused(self, tmp_path):
        path = tmp_path / "f.txt"
        path.write_text("I10K 0.1\nI10K 0.2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate mutant keys"):
            R.load_frequency_file(path)

    @pytest.mark.parametrize("value", ["0.0", "-0.1"])
    def test_non_positive_frequencies_are_refused(self, tmp_path, value):
        path = tmp_path / "f.txt"
        path.write_text(f"I10K {value}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="non-positive frequencies"):
            R.load_frequency_file(path)


@pytest.mark.unit
class TestBuildLog10FrequencyMatrix:

    def test_matched_cells_carry_log10_of_the_frequency(self, parsed_matrix):
        matrix, report = R.build_log10_frequency_matrix({"I10K": 0.1, "F15D": 0.01}, parsed_matrix)
        assert matrix.at["K", 10] == pytest.approx(-1.0, abs=1e-15)
        assert matrix.at["D", 15] == pytest.approx(-2.0, abs=1e-15)
        assert report == {
            "n_frequency_records": 2, "n_matched": 2,
            "n_unmatched": 0, "unmatched_examples": [],
        }

    def test_unmatched_cells_keep_the_no_frequency_sentinel(self, parsed_matrix):
        matrix, _ = R.build_log10_frequency_matrix({"I10K": 0.1}, parsed_matrix)
        assert R.NO_FREQUENCY_SENTINEL == 999.0
        values = matrix.to_numpy()
        assert (values == R.NO_FREQUENCY_SENTINEL).sum() == values.size - 1

    @pytest.mark.parametrize(
        "mutant, why",
        [
            ("nonsense", "does not match <WT><pos><MUT>"),
            ("i10k", "lower case fails the [A-Z] regex"),
            ("M0K", "position below 1"),
            ("M999K", "position beyond the reference"),
            ("A10K", "wild-type letter disagrees with the ESCOTT column"),
            ("I10I", "self-mutation"),
            ("I10B", "B is not one of the 20 rows"),
        ],
    )
    def test_every_way_a_record_can_fail_to_land(self, parsed_matrix, mutant, why):
        matrix, report = R.build_log10_frequency_matrix({mutant: 0.1}, parsed_matrix)
        assert report["n_matched"] == 0, why
        assert report["n_unmatched"] == 1
        assert report["unmatched_examples"] == [mutant]
        assert (matrix.to_numpy() == R.NO_FREQUENCY_SENTINEL).all()

    def test_unmatched_records_are_warned_about_on_stdout(self, parsed_matrix, capsys):
        R.build_log10_frequency_matrix({"A10K": 0.1}, parsed_matrix)
        assert "did not match the ESCOTT frame" in capsys.readouterr().out

    def test_unmatched_examples_are_capped_at_ten(self, parsed_matrix):
        frequencies = {f"A{i}K": 0.1 for i in range(200, 215)}
        _, report = R.build_log10_frequency_matrix(frequencies, parsed_matrix)
        assert report["n_unmatched"] == 15
        assert len(report["unmatched_examples"]) == 10

    def test_the_production_frequency_file_lands_completely(
        self, parsed_matrix, frequency_file_factory
    ):
        frequencies = R.load_frequency_file(frequency_file_factory())
        _, report = R.build_log10_frequency_matrix(frequencies, parsed_matrix)
        assert report["n_matched"] == len(C.EXPECTED_FREQUENCY_FILE_MIN_COUNT_1)
        assert report["n_unmatched"] == 0

    def test_it_needs_the_wt_sequence_attr(self):
        with pytest.raises(ValueError, match="wt_sequence"):
            R.build_log10_frequency_matrix({}, pd.DataFrame(np.zeros((20, 3))))


@pytest.mark.unit
class TestEscottRankScores:
    """``1 - rank/N``: PRESCOTT's own space, and it FLIPS the sign convention."""

    def test_range_and_orientation(self, parsed_matrix):
        ranked = R.escott_rank_scores(parsed_matrix)
        values = ranked.to_numpy()
        assert values.min() >= 0.0 and values.max() < 1.0
        filled = R.fill_wildtype(parsed_matrix)
        best = np.unravel_index(np.argmax(filled.to_numpy()), filled.shape)
        worst = np.unravel_index(np.argmin(filled.to_numpy()), filled.shape)
        # high raw ESCOTT == tolerated == LOW prescott-space score
        assert values[best] < values[worst]

    def test_it_is_a_monotone_decreasing_function_of_the_raw_value(self, parsed_matrix):
        filled = R.fill_wildtype(parsed_matrix).to_numpy().ravel()
        ranked = R.escott_rank_scores(parsed_matrix).to_numpy().ravel()
        order = np.argsort(filled)
        assert np.all(np.diff(ranked[order]) <= 1e-12)

    def test_full_precision_is_kept(self, escott_matrix_factory):
        """prescott.py:952 formats with '{:6.2f}', collapsing ~11k values to ~97.

        The fixture matrix repeats the same 20 values in every column, so this
        needs a matrix whose cells are genuinely distinct.
        """
        values = {}
        for position, wt in enumerate(C.QUERY_PROTEIN, start=1):
            for index, aa in enumerate(C.ESCOTT_ROW_ORDER):
                upper = aa.upper()
                values[(upper, position)] = None if upper == wt else -(index * 100 + position) / 1000.0
        path, _ = escott_matrix_factory(values=values)
        ranked = R.escott_rank_scores(R.read_escott_matrix(path))
        distinct = len(np.unique(np.round(ranked.to_numpy(), 12)))
        assert distinct > 500
        # ... and two-decimal quantisation would have destroyed almost all of it
        assert len(np.unique(np.round(ranked.to_numpy(), 2))) < distinct / 4

    def test_the_fill_mode_is_honoured(self, parsed_matrix):
        column = R.escott_rank_scores(parsed_matrix, wildtype_fill="column_max")
        global_ = R.escott_rank_scores(parsed_matrix, wildtype_fill="global_max")
        assert not np.allclose(column.to_numpy(), global_.to_numpy())


@pytest.mark.unit
class TestApplyPrescottEquation:
    """Hand-checked arithmetic; see prescott.py:747-819 for each branch."""

    # r = 0.5, Fc = -2.0, c = 0.4
    #   above cutoff  f = -0.5:  eq1 -> 0.5 - (-0.5*0.4/-2.0)      = 0.4
    #                            eq2 -> 0.5 - 0.4*(-2+0.5)/(-2)    = 0.2
    #                            eq3 -> same as eq2                = 0.2
    #                            eq5 -> 0.0
    #   below cutoff  f = -3.0:  eq1/eq2/eq5 -> untouched          = 0.5
    #                            eq3 -> min(0.5 - 0.4*(-2+3)/(-2), 1) = 0.7
    RANKED = small_frame([[0.5, 0.5]])
    ABOVE = small_frame([[-0.5, R.NO_FREQUENCY_SENTINEL]])
    BELOW = small_frame([[-3.0, R.NO_FREQUENCY_SENTINEL]])

    @pytest.mark.parametrize("equation, expected", [(1, 0.4), (2, 0.2), (3, 0.2), (5, 0.0)])
    def test_above_cutoff_branches(self, equation, expected):
        out = R.apply_prescott_equation(self.RANKED, self.ABOVE, 0.4, -2.0, equation=equation)
        assert out.iat[0, 0] == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize("equation, expected", [(1, 0.5), (2, 0.5), (3, 0.7), (5, 0.5)])
    def test_below_cutoff_only_equation_three_acts(self, equation, expected):
        out = R.apply_prescott_equation(self.RANKED, self.BELOW, 0.4, -2.0, equation=equation)
        assert out.iat[0, 0] == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize("equation", R.SUPPORTED_PRESCOTT_EQUATIONS)
    def test_the_sentinel_cell_is_never_touched(self, equation):
        out = R.apply_prescott_equation(self.RANKED, self.ABOVE, 0.4, -2.0, equation=equation)
        assert out.iat[0, 1] == 0.5

    def test_equation_two_clips_at_zero(self):
        ranked = small_frame([[0.05, 0.05]])
        out = R.apply_prescott_equation(ranked, self.ABOVE, 0.4, -2.0, equation=2)
        assert out.iat[0, 0] == 0.0                     # -0.25 clipped, not negative

    def test_equation_three_caps_at_one(self):
        ranked = small_frame([[0.9, 0.9]])
        out = R.apply_prescott_equation(ranked, self.BELOW, 0.4, -2.0, equation=3)
        assert out.iat[0, 0] == pytest.approx(1.0, abs=1e-12)   # 1.1 capped

    def test_a_cell_exactly_at_the_cutoff_counts_as_below(self):
        at_cutoff = small_frame([[-2.0, R.NO_FREQUENCY_SENTINEL]])
        assert R.apply_prescott_equation(
            self.RANKED, at_cutoff, 0.4, -2.0, equation=2
        ).iat[0, 0] == 0.5

    def test_coefficient_zero_is_the_identity(self, parsed_matrix, frequency_file_factory):
        ranked = R.escott_rank_scores(parsed_matrix)
        log10_frequency, _ = R.build_log10_frequency_matrix(
            R.load_frequency_file(frequency_file_factory()), parsed_matrix
        )
        out = R.apply_prescott_equation(ranked, log10_frequency, 0.0, -2.0, equation=2)
        assert np.allclose(out.to_numpy(), ranked.to_numpy(), atol=0.0)

    def test_equation_four_is_refused(self):
        with pytest.raises(ValueError, match="equation 4 is not supported"):
            R.apply_prescott_equation(self.RANKED, self.ABOVE, 0.4, -2.0, equation=4)

    @pytest.mark.parametrize("equation", [0, 6, -1])
    def test_other_unknown_equations_are_refused(self, equation):
        with pytest.raises(ValueError, match="is not supported"):
            R.apply_prescott_equation(self.RANKED, self.ABOVE, 0.4, -2.0, equation=equation)

    @pytest.mark.parametrize("cutoff", [0.0, 1.0, 4.0])
    def test_a_non_negative_cutoff_is_refused(self, cutoff):
        with pytest.raises(ValueError, match="cutoff must be negative"):
            R.apply_prescott_equation(self.RANKED, self.ABOVE, 0.4, cutoff, equation=2)

    def test_the_input_frame_is_not_mutated(self):
        ranked = small_frame([[0.5, 0.5]])
        R.apply_prescott_equation(ranked, self.ABOVE, 0.4, -2.0, equation=5)
        assert ranked.iat[0, 0] == 0.5

    def test_labels_and_attrs_survive(self, parsed_matrix, frequency_file_factory):
        ranked = R.escott_rank_scores(parsed_matrix)
        log10_frequency, _ = R.build_log10_frequency_matrix(
            R.load_frequency_file(frequency_file_factory()), parsed_matrix
        )
        out = R.apply_prescott_equation(ranked, log10_frequency, 0.5, -2.0)
        assert list(out.index) == list(C.PLM_CACHE_ROW_ORDER)
        assert out.attrs["wt_sequence"] == C.QUERY_PROTEIN


@pytest.mark.unit
class TestPrescottV2Scores:

    def test_coefficient_zero_is_the_exact_identity(self, parsed_matrix, frequency_file_factory):
        """The property that makes ESCOTT-vs-PRESCOTT an honest ablation."""
        log10_frequency, _ = R.build_log10_frequency_matrix(
            R.load_frequency_file(frequency_file_factory()), parsed_matrix
        )
        remapped = R.prescott_v2_scores(parsed_matrix, log10_frequency, 0.0, -2.0)
        filled = R.fill_wildtype(parsed_matrix)
        assert np.allclose(remapped.to_numpy(), filled.to_numpy(), atol=1e-12)

    def test_coefficient_zero_identity_survives_a_tie_free_matrix(
        self, escott_matrix_factory, frequency_file_factory
    ):
        """The default fixture repeats 20 values per column, so its rank plateaus
        hide an off-by-one in the quantile remap.  This matrix has 1440 distinct
        cells, which is what makes the ``(A*N - 1)/(N - 1)`` correction visible:
        indexing with ``A`` directly leaves a systematic ~1/N shift.
        """
        values = {}
        for position, wt in enumerate(C.QUERY_PROTEIN, start=1):
            for index, aa in enumerate(C.ESCOTT_ROW_ORDER):
                upper = aa.upper()
                values[(upper, position)] = None if upper == wt else -(index * 100 + position) / 1000.0
        path, _ = escott_matrix_factory(values=values)
        matrix = R.read_escott_matrix(path)
        log10_frequency, _ = R.build_log10_frequency_matrix(
            R.load_frequency_file(frequency_file_factory()), matrix
        )
        remapped = R.prescott_v2_scores(matrix, log10_frequency, 0.0, -2.0)
        filled = R.fill_wildtype(matrix)
        assert np.allclose(remapped.to_numpy(), filled.to_numpy(), atol=1e-12)

    def test_only_frequency_carrying_cells_move(self, parsed_matrix, frequency_file_factory):
        log10_frequency, _ = R.build_log10_frequency_matrix(
            R.load_frequency_file(frequency_file_factory()), parsed_matrix
        )
        remapped = R.prescott_v2_scores(parsed_matrix, log10_frequency, 0.5, -2.0)
        filled = R.fill_wildtype(parsed_matrix)
        moved = np.abs(remapped.to_numpy() - filled.to_numpy()) > 1e-9
        has_frequency = log10_frequency.to_numpy() != R.NO_FREQUENCY_SENTINEL
        assert moved.sum() > 0
        assert not (moved & ~has_frequency).any()

    def test_the_result_stays_on_the_raw_escott_scale(self, parsed_matrix, frequency_file_factory):
        """Rank-space numbers in the softmax would make one alpha grid meaningless."""
        log10_frequency, _ = R.build_log10_frequency_matrix(
            R.load_frequency_file(frequency_file_factory()), parsed_matrix
        )
        remapped = R.prescott_v2_scores(parsed_matrix, log10_frequency, 1.0, -2.0)
        filled = R.fill_wildtype(parsed_matrix).to_numpy()
        assert remapped.to_numpy().min() >= filled.min() - 1e-12
        assert remapped.to_numpy().max() <= filled.max() + 1e-12

    def test_a_penalised_mutant_moves_towards_tolerated(
        self, parsed_matrix, frequency_file_factory
    ):
        """Equation 2 penalises over-frequent variants, i.e. declares them benign,
        which on the raw ESCOTT scale means a HIGHER (less negative) value."""
        log10_frequency, _ = R.build_log10_frequency_matrix(
            R.load_frequency_file(frequency_file_factory()), parsed_matrix
        )
        remapped = R.prescott_v2_scores(parsed_matrix, log10_frequency, 1.0, -2.0, equation=2)
        filled = R.fill_wildtype(parsed_matrix)
        assert remapped.at["D", 35] > filled.at["D", 35]

    def test_the_fill_mode_is_forwarded(self, parsed_matrix, frequency_file_factory):
        log10_frequency, _ = R.build_log10_frequency_matrix(
            R.load_frequency_file(frequency_file_factory()), parsed_matrix
        )
        column = R.prescott_v2_scores(parsed_matrix, log10_frequency, 0.0, -2.0)
        global_ = R.prescott_v2_scores(
            parsed_matrix, log10_frequency, 0.0, -2.0, wildtype_fill="global_max"
        )
        assert not np.allclose(column.to_numpy(), global_.to_numpy())

    def test_labels_and_attrs_survive(self, parsed_matrix, frequency_file_factory):
        log10_frequency, _ = R.build_log10_frequency_matrix(
            R.load_frequency_file(frequency_file_factory()), parsed_matrix
        )
        out = R.prescott_v2_scores(parsed_matrix, log10_frequency, 0.5, -2.0)
        assert list(out.index) == list(C.PLM_CACHE_ROW_ORDER)
        assert list(out.columns) == list(parsed_matrix.columns)
        assert out.attrs["wt_sequence"] == C.QUERY_PROTEIN


@pytest.mark.unit
class TestCountClippedToZero:

    def test_counts_only_frequency_carrying_cells_driven_to_zero(self):
        ranked = small_frame([[0.05, 0.05, 0.0]], columns=(1, 2, 3))
        log10_frequency = small_frame([[-0.5, R.NO_FREQUENCY_SENTINEL, -0.5]], columns=(1, 2, 3))
        penalised = R.apply_prescott_equation(ranked, log10_frequency, 0.4, -2.0, equation=2)
        report = R.count_clipped_to_zero(ranked, penalised, log10_frequency)
        # column 1 clipped; column 2 has no frequency; column 3 was already 0
        assert report == {"n_mutants_with_frequency": 2, "n_clipped_to_zero": 1}

    def test_nothing_is_clipped_at_coefficient_zero(self, parsed_matrix, frequency_file_factory):
        log10_frequency, _ = R.build_log10_frequency_matrix(
            R.load_frequency_file(frequency_file_factory()), parsed_matrix
        )
        ranked = R.escott_rank_scores(parsed_matrix)
        penalised = R.apply_prescott_equation(ranked, log10_frequency, 0.0, -2.0)
        report = R.count_clipped_to_zero(ranked, penalised, log10_frequency)
        assert report["n_mutants_with_frequency"] == len(C.EXPECTED_FREQUENCY_FILE_MIN_COUNT_1)
        assert report["n_clipped_to_zero"] == 0


# =========================================================================== #
# 7. Score-matrix emission (the plm_cache contract)
# =========================================================================== #

@pytest.mark.unit
class TestWriteScoreMatrix:

    def test_layout_is_the_plm_cache_layout(self, tmp_path, parsed_matrix):
        probabilities = R.escott_to_probability(parsed_matrix)
        out = R.write_score_matrix(probabilities, C.QUERY_PROTEIN, tmp_path / "s.csv")
        lines = out.read_text().splitlines()
        assert len(lines) == 21
        assert lines[0].split(",") == ["sequence"] + list(C.QUERY_PROTEIN)
        assert [line.split(",")[0] for line in lines[1:]] == list(C.PLM_CACHE_ROW_ORDER)
        # header=False: the column labels 1..L must NOT appear anywhere
        assert not out.read_text().startswith(",1,2,3")

    def test_it_is_byte_identical_to_the_sc2_writer(self, tmp_path, parsed_matrix):
        """run_mutational_accessibility.py:1032-1043, reproduced literally here."""
        probabilities = R.escott_to_probability(parsed_matrix)
        ours = R.write_score_matrix(probabilities, C.QUERY_PROTEIN, tmp_path / "ours.csv")
        sequence_row = pd.DataFrame(
            [list(C.QUERY_PROTEIN)], index=["sequence"], columns=list(probabilities.columns)
        )
        probability_rows = pd.DataFrame(
            probabilities.to_numpy(),
            index=list(C.PLM_CACHE_ROW_ORDER),
            columns=list(probabilities.columns),
        )
        theirs = tmp_path / "theirs.csv"
        pd.concat([sequence_row, probability_rows], axis=0).to_csv(theirs, header=False)
        assert ours.read_bytes() == theirs.read_bytes()

    def test_rows_are_reordered_into_plm_order_when_needed(self, tmp_path, parsed_matrix):
        probabilities = R.escott_to_probability(parsed_matrix)
        shuffled = probabilities.loc[sorted(probabilities.index)]
        out = R.write_score_matrix(shuffled, C.QUERY_PROTEIN, tmp_path / "s.csv")
        lines = out.read_text().splitlines()
        assert [line.split(",")[0] for line in lines[1:]] == list(C.PLM_CACHE_ROW_ORDER)
        # ... and the values travelled with their labels
        assert float(lines[1].split(",")[1]) == pytest.approx(probabilities.at["A", 1])

    def test_a_length_mismatch_is_refused(self, tmp_path, parsed_matrix):
        probabilities = R.escott_to_probability(parsed_matrix)
        with pytest.raises(ValueError, match="does not match matrix width"):
            R.write_score_matrix(probabilities, C.QUERY_PROTEIN[:10], tmp_path / "s.csv")

    def test_the_parent_directory_is_created(self, tmp_path, parsed_matrix):
        probabilities = R.escott_to_probability(parsed_matrix)
        out = R.write_score_matrix(
            probabilities, C.QUERY_PROTEIN, tmp_path / "a" / "b" / "s.csv"
        )
        assert out.exists()

    def test_values_round_trip_at_full_precision(self, tmp_path, parsed_matrix):
        probabilities = R.escott_to_probability(parsed_matrix)
        out = R.write_score_matrix(probabilities, C.QUERY_PROTEIN, tmp_path / "s.csv")
        raw = pd.read_csv(out, index_col=0, header=None)
        back = raw.iloc[1:].astype(float).to_numpy()
        assert np.array_equal(back, probabilities.to_numpy())

    @pytest.mark.requires_rma
    def test_the_sc2_loader_reads_it_with_no_new_code(self, tmp_path, parsed_matrix):
        import run_mutational_accessibility as rma

        probabilities = R.escott_to_probability(parsed_matrix)
        out = R.write_score_matrix(probabilities, C.QUERY_PROTEIN, tmp_path / "s.csv")
        raw = pd.read_csv(out, index_col=0, header=None)
        assert rma.infer_plm_source_sequence(raw) == C.QUERY_PROTEIN
        normalised = rma.normalise_plm_matrix(raw)
        assert normalised.shape == (20, C.QUERY_LENGTH)
        assert list(normalised.index) == sorted(C.PLM_CACHE_ROW_ORDER)
        assert np.allclose(
            normalised.to_numpy(),
            probabilities.loc[list(normalised.index)].to_numpy(),
            atol=0.0,
        )


@pytest.mark.unit
class TestWriteRawMatrix:

    def test_tsv_dump_round_trips(self, tmp_path, parsed_matrix):
        out = R.write_raw_matrix(parsed_matrix, tmp_path / "raw" / "m.tsv")
        back = pd.read_csv(out, sep="\t", index_col=0)
        back.columns = [int(c) for c in back.columns]
        assert list(back.index) == list(C.PLM_CACHE_ROW_ORDER)
        finite = ~parsed_matrix.isna().to_numpy()
        assert np.allclose(
            back.to_numpy()[finite], parsed_matrix.to_numpy()[finite], atol=1e-9
        )
        assert back.isna().sum().sum() == C.QUERY_LENGTH   # the wild-type cells


# =========================================================================== #
# 8. The published prescott tool: reference run and parity check
# =========================================================================== #

def make_details_frame(
    matrix: pd.DataFrame,
    log10_frequency: pd.DataFrame,
    coefficient: float,
    cutoff: float,
    equation: int = 2,
    wildtype_fill: str = "global_max",
) -> pd.DataFrame:
    """A stand-in for ``<out>-details.csv``, quantised to 2 dp like the real one.

    The numbers come from the module's own arithmetic on purpose: the parity
    check's job in THIS test is the plumbing (mutant matching, rounding, the
    wild-type fill choice, the pass/fail verdict), and the arithmetic itself is
    checked against the real ``prescott`` binary in :class:`TestRealPrescott`.
    """
    ranked = R.escott_rank_scores(matrix, wildtype_fill=wildtype_fill)
    penalised = R.apply_prescott_equation(
        ranked, log10_frequency, coefficient, cutoff, equation=equation
    )
    protein = R.escott_wt_sequence(matrix)
    rows = []
    for position, wt in enumerate(protein, start=1):
        for aa in C.PLM_CACHE_ROW_ORDER:
            if aa == wt:
                continue
            rows.append({
                "mutant": f"{wt}{position}{aa}",
                "ESCOTT": round(float(ranked.at[aa, position]), 2),
                "protein": "HAK",
                "log10frequency": float(log10_frequency.at[aa, position]),
                "labels": "",
                "position": position,
                "Selected Population": "",
                "PRESCOTT": round(float(penalised.at[aa, position]), 2),
            })
    return pd.DataFrame(rows)


@pytest.fixture()
def prescott_case(tmp_path, parsed_matrix, frequency_file_factory):
    """Everything the reference run and the parity check need, in one place."""
    escott_txt = Path(parsed_matrix.attrs["source_path"])
    query = C.write_fasta(tmp_path / "K_query.fasta", [(C.QUERY_HEADER, C.QUERY_PROTEIN)])
    frequency = frequency_file_factory(name="K_parent_frequency.txt")
    log10_frequency, _ = R.build_log10_frequency_matrix(
        R.load_frequency_file(frequency), parsed_matrix
    )
    return {
        "escott_txt": escott_txt,
        "query": query,
        "frequency": frequency,
        "matrix": parsed_matrix,
        "log10_frequency": log10_frequency,
        "out_dir": tmp_path / "prescott_ref" / "K",
        "details": make_details_frame(parsed_matrix, log10_frequency, 0.5, -2.0, 2),
    }


@pytest.mark.unit
class TestRunPrescottReference:

    def _run(self, case, fake_tools_factory=None, **kwargs):
        return R.run_prescott_reference(
            kwargs.pop("lineage", "K"),
            case["escott_txt"], case["query"], case["frequency"], case["out_dir"],
            coefficient=kwargs.pop("coefficient", 0.5),
            frequency_cutoff=kwargs.pop("frequency_cutoff", -2.0),
            **kwargs,
        )

    def test_command_and_staging(self, prescott_case, fake_tools):
        recorder = fake_tools(details_rows=prescott_case["details"])
        details = self._run(prescott_case)
        assert details.name == "prescott_K-details.csv"
        call = recorder.prescott_calls[0]
        assert call["cmd"] == [
            str(R.DEFAULT_PRESCOTT_BIN),
            "-e", prescott_case["escott_txt"].name,
            "-s", "K_query.fasta",
            "-g", "K_parent_frequency.txt",
            "-o", "prescott_K",
            "--equation", "2",
            "-c", "0.5",
            "-f", "-2.0",
            "--escottformat", "gemme",
        ]
        assert call["cwd"] == prescott_case["out_dir"]
        # bare filenames again, and every input staged beside the outputs
        for name in (prescott_case["escott_txt"].name, "K_query.fasta",
                     "K_parent_frequency.txt"):
            assert (prescott_case["out_dir"] / name).exists()

    def test_usefrequencies_false_is_never_passed(self, prescott_case, fake_tools):
        """prescott.py:1103-1107 would leave 'position' empty and IndexError."""
        recorder = fake_tools(details_rows=prescott_case["details"])
        self._run(prescott_case)
        assert "--usefrequencies" not in recorder.prescott_calls[0]["cmd"]

    def test_the_output_stem_is_dot_free(self, prescott_case, fake_tools):
        """prescott.py:902 runs splitext on -o, truncating at the LAST dot."""
        recorder = fake_tools(details_rows=prescott_case["details"])
        R.run_prescott_reference(
            "J.2_int", prescott_case["escott_txt"], prescott_case["query"],
            prescott_case["frequency"], prescott_case["out_dir"],
            coefficient=0.5, frequency_cutoff=-2.0,
        )
        cmd = recorder.prescott_calls[0]["cmd"]
        stem = cmd[cmd.index("-o") + 1]
        assert stem == "prescott_J_2_int"
        assert "." not in stem

    def test_a_csv_frequency_file_is_refused(self, prescott_case, tmp_path, fake_tools):
        """prescott.py:982 switches to the gnomAD parser on that suffix."""
        fake_tools(details_rows=prescott_case["details"])
        bad = tmp_path / "K_parent_frequency.csv"
        shutil.copy2(prescott_case["frequency"], bad)
        with pytest.raises(ValueError, match="gnomAD parser"):
            R.run_prescott_reference(
                "K", prescott_case["escott_txt"], prescott_case["query"], bad,
                prescott_case["out_dir"], coefficient=0.5, frequency_cutoff=-2.0,
            )

    def test_a_query_matrix_width_mismatch_is_caught_before_the_subprocess(
        self, prescott_case, tmp_path, fake_tools
    ):
        recorder = fake_tools(details_rows=prescott_case["details"])
        short = C.write_fasta(tmp_path / "short_query.fasta", [("HAK", C.QUERY_PROTEIN[:10])])
        with pytest.raises(ValueError, match="prescott would IndexError"):
            R.run_prescott_reference(
                "K", prescott_case["escott_txt"], short, prescott_case["frequency"],
                prescott_case["out_dir"], coefficient=0.5, frequency_cutoff=-2.0,
            )
        assert recorder.prescott_calls == []

    def test_a_nonzero_return_code_is_a_failure(self, prescott_case, fake_tools):
        fake_tools(details_rows=prescott_case["details"], prescott_returncode=3)
        with pytest.raises(RuntimeError, match="prescott failed \\(rc=3\\)"):
            self._run(prescott_case)

    def test_a_missing_details_file_is_a_failure_even_at_rc_zero(
        self, prescott_case, fake_tools
    ):
        fake_tools(write_details=False)
        with pytest.raises(RuntimeError, match="prescott failed"):
            self._run(prescott_case)

    def test_the_log_is_captured(self, prescott_case, fake_tools):
        fake_tools(details_rows=prescott_case["details"])
        self._run(prescott_case)
        assert (prescott_case["out_dir"] / "prescott_K.log").exists()

    def test_identical_inputs_are_cached(self, prescott_case, fake_tools, capsys):
        recorder = fake_tools(details_rows=prescott_case["details"])
        self._run(prescott_case)
        capsys.readouterr()
        self._run(prescott_case)
        assert len(recorder.prescott_calls) == 1
        assert "cache hit" in capsys.readouterr().out

    def test_a_changed_coefficient_busts_the_cache(self, prescott_case, fake_tools):
        recorder = fake_tools(details_rows=prescott_case["details"])
        self._run(prescott_case)
        self._run(prescott_case, coefficient=1.0)
        assert len(recorder.prescott_calls) == 2

    def test_a_changed_frequency_file_busts_the_cache(self, prescott_case, fake_tools):
        recorder = fake_tools(details_rows=prescott_case["details"])
        self._run(prescott_case)
        C.write_frequency_file(prescott_case["frequency"], {"I10K": 0.5})
        self._run(prescott_case)
        assert len(recorder.prescott_calls) == 2

    def test_force_ignores_the_cache(self, prescott_case, fake_tools):
        recorder = fake_tools(details_rows=prescott_case["details"])
        self._run(prescott_case)
        self._run(prescott_case, force=True)
        assert len(recorder.prescott_calls) == 2

    def test_a_corrupt_marker_busts_the_cache(self, prescott_case, fake_tools):
        recorder = fake_tools(details_rows=prescott_case["details"])
        self._run(prescott_case)
        (prescott_case["out_dir"] / "prescott_K_exit.json").write_text("{", encoding="utf-8")
        self._run(prescott_case)
        assert len(recorder.prescott_calls) == 2


@pytest.mark.unit
class TestReadPrescottDetails:

    def test_reads_and_upper_cases(self, tmp_path):
        frame = pd.DataFrame([{"mutant": "i10k", "ESCOTT": 0.5, "PRESCOTT": 0.2,
                               "log10frequency": -1.0}])
        path = tmp_path / "d.csv"
        frame.to_csv(path, index=False)
        assert R.read_prescott_details(path)["mutant"].tolist() == ["I10K"]

    def test_missing_columns_are_reported_by_name(self, tmp_path):
        path = tmp_path / "d.csv"
        pd.DataFrame([{"mutant": "I10K", "ESCOTT": 0.5}]).to_csv(path, index=False)
        with pytest.raises(ValueError, match=r"missing expected columns \['PRESCOTT', 'log10frequency'\]"):
            R.read_prescott_details(path)


@pytest.mark.unit
class TestPrescottParityCheck:

    def test_matching_numbers_pass_with_zero_delta(self, prescott_case):
        report = R.prescott_parity_check(
            prescott_case["matrix"], prescott_case["log10_frequency"],
            prescott_case["details"], coefficient=0.5, frequency_cutoff=-2.0,
        )
        assert report.attrs["passed"] is True
        assert report.attrs["max_abs_delta_escott"] == 0.0
        assert report.attrs["max_abs_delta_prescott"] == 0.0
        # 72 positions x 19 non-wild-type residues
        assert len(report) == C.QUERY_LENGTH * 19
        assert list(report.columns) == [
            "mutant", "ours_escott", "theirs_escott", "ours_prescott", "theirs_prescott",
            "abs_delta_escott", "abs_delta_prescott",
        ]

    def test_a_disagreement_beyond_tolerance_fails(self, prescott_case):
        details = prescott_case["details"].copy()
        details.loc[0, "PRESCOTT"] = details.loc[0, "PRESCOTT"] + 0.5
        report = R.prescott_parity_check(
            prescott_case["matrix"], prescott_case["log10_frequency"], details,
            coefficient=0.5, frequency_cutoff=-2.0,
        )
        assert report.attrs["passed"] is False
        assert report.attrs["max_abs_delta_prescott"] == pytest.approx(0.5, abs=1e-9)

    def test_a_rounding_sized_disagreement_still_passes(self, prescott_case):
        details = prescott_case["details"].copy()
        details.loc[0, "PRESCOTT"] = details.loc[0, "PRESCOTT"] + 0.01
        report = R.prescott_parity_check(
            prescott_case["matrix"], prescott_case["log10_frequency"], details,
            coefficient=0.5, frequency_cutoff=-2.0,
        )
        assert report.attrs["passed"] is True

    def test_the_tolerance_is_configurable(self, prescott_case):
        details = prescott_case["details"].copy()
        details.loc[0, "ESCOTT"] = details.loc[0, "ESCOTT"] + 0.01
        assert R.prescott_parity_check(
            prescott_case["matrix"], prescott_case["log10_frequency"], details,
            coefficient=0.5, frequency_cutoff=-2.0, tolerance=0.001,
        ).attrs["passed"] is False

    def test_it_uses_the_global_max_fill_prescott_itself_uses(self, escott_matrix_factory):
        """prescott.py:1030 fills with np.nanmax over the WHOLE matrix.  Using our
        production column_max fill here would be a spurious mismatch.

        The default fixture cannot see the difference (its widest non-wild-type
        disagreement is 0.0024, inside the 0.011 tolerance), so this uses a matrix
        with one wildly out-of-scale column: the two fills then put that column's
        wild-type cell at opposite ends of the global ordering and the rounded
        ranks differ by 0.02.
        """
        protein = "MKT"
        values = {}
        for position, wt in enumerate(protein, start=1):
            for index, aa in enumerate(C.ESCOTT_ROW_ORDER):
                upper = aa.upper()
                values[(upper, position)] = None if upper == wt else (
                    -100.0 - index if position == 1 else -index * 0.01
                )
        path, _ = escott_matrix_factory(protein=protein, values=values)
        matrix = R.read_escott_matrix(path)
        log10_frequency, _ = R.build_log10_frequency_matrix({"M1K": 0.1}, matrix)

        honest = make_details_frame(matrix, log10_frequency, 0.5, -2.0, 2,
                                    wildtype_fill="global_max")
        assert R.prescott_parity_check(
            matrix, log10_frequency, honest, coefficient=0.5, frequency_cutoff=-2.0
        ).attrs["passed"] is True

        wrong_fill = make_details_frame(matrix, log10_frequency, 0.5, -2.0, 2,
                                        wildtype_fill="column_max")
        report = R.prescott_parity_check(
            matrix, log10_frequency, wrong_fill, coefficient=0.5, frequency_cutoff=-2.0
        )
        assert report.attrs["passed"] is False
        assert report.attrs["max_abs_delta_escott"] == pytest.approx(0.02, abs=1e-9)

    def test_wild_type_self_mutations_are_skipped(self, prescott_case):
        details = pd.concat(
            [prescott_case["details"],
             pd.DataFrame([{"mutant": "M1M", "ESCOTT": 99.0, "protein": "HAK",
                            "log10frequency": 999.0, "labels": "", "position": 1,
                            "Selected Population": "", "PRESCOTT": 99.0}])],
            ignore_index=True,
        )
        report = R.prescott_parity_check(
            prescott_case["matrix"], prescott_case["log10_frequency"], details,
            coefficient=0.5, frequency_cutoff=-2.0,
        )
        assert "M1M" not in set(report["mutant"])
        assert report.attrs["passed"] is True

    @pytest.mark.parametrize("mutant", ["nonsense", "A1C", "M999K"])
    def test_uncomparable_rows_are_skipped(self, prescott_case, mutant):
        details = pd.DataFrame([{"mutant": mutant, "ESCOTT": 0.5, "protein": "HAK",
                                 "log10frequency": 999.0, "labels": "", "position": 1,
                                 "Selected Population": "", "PRESCOTT": 0.5}])
        with pytest.raises(ValueError, match="no comparable mutants"):
            R.prescott_parity_check(
                prescott_case["matrix"], prescott_case["log10_frequency"], details,
                coefficient=0.5, frequency_cutoff=-2.0,
            )


# =========================================================================== #
# 9. Variant naming
# =========================================================================== #

@pytest.mark.unit
class TestBuildVariantName:

    @pytest.mark.parametrize(
        "equation, coefficient, k, parent, expected",
        [
            (2, 0.5, 1, "J.2.4", "PRESCOTT_eq2_c0p50_k1_parentJ24"),
            (2, 0.5, 1, "J.2_int", "PRESCOTT_eq2_c0p50_k1_parentJ2int"),
            (3, 1.0, 2, "G.1", "PRESCOTT_eq3_c1p00_k2_parentG1"),
            (1, 0.25, 1, "J_int", "PRESCOTT_eq1_c0p25_k1_parentJint"),
            (5, 2.0, 3, "K", "PRESCOTT_eq5_c2p00_k3_parentK"),
            (2, 0.5, 1, "NA", "PRESCOTT_eq2_c0p50_k1_parentNA"),
        ],
    )
    def test_names(self, equation, coefficient, k, parent, expected):
        assert R.build_variant_name(equation, coefficient, k, parent) == expected

    def test_the_name_is_dot_free_and_filesystem_safe(self):
        name = R.build_variant_name(2, 0.5, 1, "J.2_int")
        assert "." not in name and "/" not in name and " " not in name

    def test_the_parent_token_comes_from_constants_not_a_local_rederivation(self):
        """Stage A names ``K_parentJ2int_frequency.txt`` with the same function;
        a one-character drift silently reclassifies every sensitivity variant."""
        for parent in ("J.2.4", "J.2_int", "G.1", "J_int", "K"):
            token = constants.variant_parent_token(parent)
            assert R.build_variant_name(2, 0.5, 1, parent).endswith(f"_parent{token}")
        assert R.variant_parent_token is constants.variant_parent_token

    def test_the_contested_edge_gets_two_distinguishable_names(self):
        child, primary, sensitivity = C.CONTESTED_EDGE
        assert child == "K"
        first = R.build_variant_name(2, 0.5, 1, primary)
        second = R.build_variant_name(2, 0.5, 1, sensitivity)
        assert first != second
        assert first.endswith("_parentJ24") and second.endswith("_parentJ2int")

    def test_the_coefficient_is_always_two_decimals(self):
        assert R.build_variant_name(2, 0.5, 1, "K") == R.build_variant_name(2, 0.50, 1, "K")
        assert "c0p50" in R.build_variant_name(2, 0.5, 1, "K")
        assert "c10p00" in R.build_variant_name(2, 10.0, 1, "K")


# =========================================================================== #
# 10. Fc resolution
# =========================================================================== #

@pytest.fixture()
def frequency_report_tree(tmp_path: Path) -> Path:
    """An inputs dir carrying stage A's ``frequency/frequency_report.json``."""
    inputs = tmp_path / "inputs"
    (inputs / "frequency").mkdir(parents=True)
    report = {
        "K": {
            "median_mapped_depth": 877.0,
            "frequency_cutoffs": {"1": -2.9430, "2": -2.6420},
            "frequency_cutoff_mode": "depth_scaled",
        },
        "K_parentJ2int_frequency": {
            "median_mapped_depth": 27452.0,
            "frequency_cutoffs": {"1": -4.4386},
            "frequency_cutoff_mode": "depth_scaled",
        },
        "J_int": {"median_mapped_depth": 4132.0},
        "J.2.4": {"median_depth": 5561.0},
        "J.2_int": {"parent_median_depth": 27452.0},
        "fixed_lineage": {
            "median_mapped_depth": 100.0,
            "frequency_cutoffs": {"1": -4.0},
            "frequency_cutoff_mode": "fixed",
        },
    }
    (inputs / "frequency" / "frequency_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    return inputs


@pytest.mark.unit
class TestResolveFrequencyCutoff:

    def test_stage_as_own_number_is_reused_verbatim(self, frequency_report_tree):
        assert R.resolve_frequency_cutoff("K", 1, frequency_report_tree, -4.0) == (-2.9430, 877.0)
        assert R.resolve_frequency_cutoff("K", 2, frequency_report_tree, -4.0) == (-2.6420, 877.0)

    def test_the_alternate_parent_gets_its_own_fc(self, frequency_report_tree):
        """Reusing the primary's Fc would compare two models under two different
        penalty scales -- K <- J.2.4 is 877 deep, K <- J.2_int is 27452."""
        primary, primary_depth = R.resolve_frequency_cutoff("K", 1, frequency_report_tree, -4.0)
        alternate, alternate_depth = R.resolve_frequency_cutoff(
            "K", 1, frequency_report_tree, -4.0, report_key="K_parentJ2int_frequency"
        )
        assert (primary, primary_depth) == (-2.9430, 877.0)
        assert (alternate, alternate_depth) == (-4.4386, 27452.0)

    def test_an_unlisted_k_is_computed_from_the_depth(self, frequency_report_tree):
        cutoff, depth = R.resolve_frequency_cutoff("K", 3, frequency_report_tree, -4.0)
        assert depth == 877.0
        assert cutoff == pytest.approx(np.log10(3 / 877.0), abs=1e-12)

    @pytest.mark.parametrize(
        "lineage, depth", [("J_int", 4132.0), ("J.2.4", 5561.0), ("J.2_int", 27452.0)]
    )
    def test_every_accepted_depth_key_is_read(self, frequency_report_tree, lineage, depth):
        cutoff, got = R.resolve_frequency_cutoff(lineage, 1, frequency_report_tree, -4.0)
        assert got == depth
        assert cutoff == pytest.approx(np.log10(1.0 / depth), abs=1e-12)

    def test_fixed_mode_returns_the_fallback_but_still_reports_the_depth(
        self, frequency_report_tree
    ):
        assert R.resolve_frequency_cutoff(
            "K", 1, frequency_report_tree, -4.0, mode="fixed"
        ) == (-4.0, 877.0)

    def test_a_cached_cutoff_from_another_mode_is_not_reused(self, frequency_report_tree):
        """The report holds a 'fixed' Fc; asking for depth_scaled must recompute."""
        cutoff, depth = R.resolve_frequency_cutoff(
            "fixed_lineage", 1, frequency_report_tree, -9.9, mode="depth_scaled"
        )
        assert depth == 100.0
        assert cutoff == pytest.approx(-2.0, abs=1e-12)

    def test_the_meta_table_is_the_fallback_source_of_depth(self, tmp_path):
        inputs = tmp_path / "inputs"
        (inputs / "frequency").mkdir(parents=True)
        C.write_frequency_meta(
            inputs / "frequency" / "K_parent_frequency_meta.tsv",
            C.EXPECTED_FREQUENCY_FILE_MIN_COUNT_1, "J.2.4",
        )
        cutoff, depth = R.resolve_frequency_cutoff("K", 1, inputs, -4.0)
        assert depth == C.PARENT_PANEL_MEDIAN_DEPTH
        assert cutoff == pytest.approx(-2.0, abs=1e-12)

    def test_the_meta_path_override_selects_the_alternate_panel(self, tmp_path):
        inputs = tmp_path / "inputs"
        (inputs / "frequency").mkdir(parents=True)
        override = inputs / "frequency" / "K_parentJ2int_frequency_meta.tsv"
        rows = ["mutant\tposition\twt\tmut\tcount\tdepth\tfrequency\tparent_lineage"]
        rows += [f"I10K\t10\tI\tK\t1\t1000\t0.001\tJ.2_int"] * 3
        override.write_text("\n".join(rows) + "\n", encoding="utf-8")
        cutoff, depth = R.resolve_frequency_cutoff(
            "K", 1, inputs, -4.0, meta_path_override=override
        )
        assert depth == 1000.0
        assert cutoff == pytest.approx(-3.0, abs=1e-12)

    def test_no_depth_anywhere_falls_back_and_warns(self, tmp_path, capsys):
        cutoff, depth = R.resolve_frequency_cutoff("K", 1, tmp_path, -4.0)
        assert (cutoff, depth) == (-4.0, None)
        assert "no parent depth available for K" in capsys.readouterr().out

    def test_the_report_key_appears_in_the_warning(self, tmp_path, capsys):
        R.resolve_frequency_cutoff("K", 1, tmp_path, -4.0, report_key="K_parentJ2int_frequency")
        assert "K_parentJ2int_frequency" in capsys.readouterr().out

    def test_a_zero_depth_is_treated_as_no_depth(self, tmp_path):
        """log10(k/0) is -inf, so a zero depth must take the fallback, not divide."""
        inputs = tmp_path / "inputs"
        (inputs / "frequency").mkdir(parents=True)
        (inputs / "frequency" / "frequency_report.json").write_text(
            json.dumps({"K": {"median_mapped_depth": 0.0}}), encoding="utf-8"
        )
        assert R.resolve_frequency_cutoff("K", 1, inputs, -4.0) == (-4.0, None)

    def test_a_non_dict_report_entry_does_not_crash(self, tmp_path):
        inputs = tmp_path / "inputs"
        (inputs / "frequency").mkdir(parents=True)
        (inputs / "frequency" / "frequency_report.json").write_text(
            json.dumps({"K": "not a dict"}), encoding="utf-8"
        )
        assert R.resolve_frequency_cutoff("K", 1, inputs, -4.0) == (-4.0, None)


# =========================================================================== #
# 11. Input resolution: lineage products, alternate parents, parent maps
# =========================================================================== #

@pytest.mark.unit
class TestResolveLineageInputs:

    def test_every_stage_a_b_product_is_located(self, prepared_inputs_tree):
        inputs_dir = Path(prepared_inputs_tree["inputs_dir"])
        resolved = R.resolve_lineage_inputs(inputs_dir, "J.2_int")
        assert resolved["lineage_key"] == "J.2_int"
        assert resolved["lineage_key_safe"] == "J_2_int"
        assert resolved["protein"] == C.QUERY_PROTEIN
        assert resolved["prot_token"] == prepared_inputs_tree["prot_tokens"]["J.2_int"]
        assert resolved["query_path"] == inputs_dir / "query" / "J.2_int_query.fasta"
        assert resolved["msa_path"] == inputs_dir / "msa" / "msa_J.2_int.fasta"
        assert resolved["jet_path"] == inputs_dir / "jet" / "J.2_int_surrogate_jet.res"
        assert resolved["frequency_path"] == \
            inputs_dir / "frequency" / "J.2_int_parent_frequency.txt"

    def test_an_input_only_lineage_has_no_frequency_file(self, prepared_inputs_tree):
        """G.1 is somebody's parent and nobody's child."""
        resolved = R.resolve_lineage_inputs(prepared_inputs_tree["inputs_dir"], "G.1")
        assert resolved["frequency_path"] is None
        assert resolved["alternate_frequency_paths"] == {}

    @pytest.mark.parametrize("missing", ["query", "msa", "jet"])
    def test_a_missing_product_names_the_path_and_the_fix(self, prepared_inputs_tree, missing):
        inputs_dir = Path(prepared_inputs_tree["inputs_dir"])
        target = {
            "query": inputs_dir / "query" / "K_query.fasta",
            "msa": inputs_dir / "msa" / "msa_K.fasta",
            "jet": inputs_dir / "jet" / "K_surrogate_jet.res",
        }[missing]
        target.unlink()
        with pytest.raises(FileNotFoundError) as excinfo:
            R.resolve_lineage_inputs(inputs_dir, "K")
        message = str(excinfo.value)
        assert str(target) in message
        assert "prepare_inputs.py" in message and "jet_surrogate.py" in message

    def test_a_missing_frequency_file_is_not_fatal(self, prepared_inputs_tree):
        inputs_dir = Path(prepared_inputs_tree["inputs_dir"])
        (inputs_dir / "frequency" / "K_parent_frequency.txt").unlink()
        assert R.resolve_lineage_inputs(inputs_dir, "K")["frequency_path"] is None


@pytest.mark.unit
class TestResolveAlternateFrequencyPaths:

    def test_the_manifest_index_is_the_preferred_source(self, prepared_inputs_tree):
        found = R.resolve_alternate_frequency_paths(prepared_inputs_tree["inputs_dir"], "K")
        child, _primary, alternate = C.CONTESTED_EDGE
        assert set(found) == {alternate}
        entry = found[alternate]
        assert entry["path"] == prepared_inputs_tree["alternate_frequency_path"]
        assert entry["report_key"] == "K_parentJ2int_frequency"
        assert Path(entry["meta_path"]).name == "K_parentJ2int_frequency_meta.tsv"

    def test_primary_edges_are_not_reported_as_alternates(self, prepared_inputs_tree):
        for label in ("J_int", "J.2_int", "J.2.4"):
            assert R.resolve_alternate_frequency_paths(
                prepared_inputs_tree["inputs_dir"], label
            ) == {}

    def test_a_manifest_entry_pointing_at_a_missing_file_warns_and_is_dropped(
        self, prepared_inputs_tree, capsys
    ):
        Path(prepared_inputs_tree["alternate_frequency_path"]).unlink()
        found = R.resolve_alternate_frequency_paths(prepared_inputs_tree["inputs_dir"], "K")
        assert found == {}
        assert "but it is missing on disk" in capsys.readouterr().out

    def test_a_corrupt_manifest_falls_through_to_the_glob(self, prepared_inputs_tree):
        Path(prepared_inputs_tree["manifest_path"]).write_text("{ not json", encoding="utf-8")
        found = R.resolve_alternate_frequency_paths(prepared_inputs_tree["inputs_dir"], "K")
        # The glob cannot invert the filename back to a label, so it reports the TOKEN.
        assert set(found) == {"J2int"}
        assert found["J2int"]["report_key"] == "K_parentJ2int_frequency"

    def test_the_glob_fallback_never_returns_the_primary_file(self, tmp_path):
        freq = tmp_path / "inputs" / "frequency"
        freq.mkdir(parents=True)
        C.write_frequency_file(freq / "K_parent_frequency.txt", {"I10K": 0.1})
        assert R.resolve_alternate_frequency_paths(tmp_path / "inputs", "K") == {}

    def test_the_glob_fallback_is_keyed_by_token_not_label(self, tmp_path):
        """Documented limitation: ``variant_parent_token`` strips dots and
        underscores, so ``K_parentJ2int_frequency.txt`` could have come from
        ``J.2_int`` or ``J2int`` and the filename alone cannot say which.  A tree
        prepared before ``frequency_index`` existed therefore cannot be scored by
        parent LABEL -- see ``test_an_unprepared_alternate_parent_is_refused``."""
        freq = tmp_path / "inputs" / "frequency"
        freq.mkdir(parents=True)
        C.write_frequency_file(freq / "K_parentJ2int_frequency.txt", {"I10K": 0.1})
        found = R.resolve_alternate_frequency_paths(tmp_path / "inputs", "K")
        assert set(found) == {"J2int"}
        assert "J.2_int" not in found

    def test_no_frequency_directory_at_all_is_empty_not_an_error(self, tmp_path):
        assert R.resolve_alternate_frequency_paths(tmp_path / "nowhere", "K") == {}


@pytest.mark.unit
class TestResolveParentMap:

    def test_the_manifest_is_the_single_source_of_truth(self, prepared_inputs_tree):
        """Stage A resolved it; stage C must not re-derive the clade topology."""
        assert R.resolve_parent_map(prepared_inputs_tree["inputs_dir"]) == C.EXPECTED_PARENT_MAP

    def test_the_default_edge_for_k_is_the_clade_evidence_one(self, prepared_inputs_tree):
        parent_map = R.resolve_parent_map(prepared_inputs_tree["inputs_dir"])
        assert parent_map["K"] == "J.2.4"
        assert parent_map["K"] != "J.2_int"

    def test_a_cli_override_patches_single_edges(self, prepared_inputs_tree):
        patched = R.resolve_parent_map(prepared_inputs_tree["inputs_dir"], "K=J.2_int")
        assert patched["K"] == "J.2_int"
        assert patched["J.2.4"] == "J.2_int"          # untouched

    def test_several_overrides_and_blank_tokens(self, prepared_inputs_tree):
        patched = R.resolve_parent_map(
            prepared_inputs_tree["inputs_dir"], "K=J.2_int, ,J.2.4=J_int"
        )
        assert patched["K"] == "J.2_int" and patched["J.2.4"] == "J_int"

    def test_no_manifest_yields_an_empty_map(self, tmp_path):
        assert R.resolve_parent_map(tmp_path) == {}

    def test_an_override_works_without_a_manifest(self, tmp_path):
        assert R.resolve_parent_map(tmp_path, "K=J.2.4") == {"K": "J.2.4"}


@pytest.mark.unit
class TestResolveSensitivityMap:

    def test_it_defaults_to_whatever_stage_a_prepared(self, prepared_inputs_tree):
        assert R.resolve_sensitivity_map(prepared_inputs_tree["inputs_dir"]) == \
            prepared_inputs_tree["sensitivity_edge"]

    def test_an_override_replaces_the_manifest_outright(self, prepared_inputs_tree):
        assert R.resolve_sensitivity_map(
            prepared_inputs_tree["inputs_dir"], "J.2.4=J_int"
        ) == {"J.2.4": "J_int"}

    def test_disable_wins_over_both(self, prepared_inputs_tree):
        assert R.resolve_sensitivity_map(
            prepared_inputs_tree["inputs_dir"], "J.2.4=J_int", disable=True
        ) == {}

    def test_no_manifest_is_empty(self, tmp_path):
        assert R.resolve_sensitivity_map(tmp_path) == {}

    def test_a_corrupt_manifest_is_empty_rather_than_fatal(self, prepared_inputs_tree):
        Path(prepared_inputs_tree["manifest_path"]).write_text("{ not json", encoding="utf-8")
        assert R.resolve_sensitivity_map(prepared_inputs_tree["inputs_dir"]) == {}

    def test_a_malformed_override_is_refused(self, prepared_inputs_tree):
        with pytest.raises(ValueError, match="malformed edge"):
            R.resolve_sensitivity_map(prepared_inputs_tree["inputs_dir"], "K")


# =========================================================================== #
# 12. process_lineage -- the whole stage, with escott mocked
# =========================================================================== #

@pytest.fixture()
def lineage_run(tmp_path, prepared_inputs_tree, fake_tools):
    """A ready-to-call ``process_lineage`` bound to the prepared tree."""
    def _run(lineage="K", **kwargs):
        kwargs.setdefault("parent_lineage", C.EXPECTED_PARENT_MAP.get(lineage))
        kwargs.setdefault("coefficients", (0.5,))
        kwargs.setdefault("equations", (2,))
        kwargs.setdefault("frequency_cutoff_ks", (1,))
        return R.process_lineage(
            lineage,
            prepared_inputs_tree["inputs_dir"],
            tmp_path / "escott",
            tmp_path / "scores",
            **kwargs,
        )
    _run.tree = prepared_inputs_tree            # type: ignore[attr-defined]
    _run.scores_dir = tmp_path / "scores"       # type: ignore[attr-defined]
    _run.escott_root = tmp_path / "escott"      # type: ignore[attr-defined]
    return _run


@pytest.mark.integration
class TestProcessLineage:

    def test_the_escott_baseline_row(self, lineage_run, fake_tools):
        fake_tools()
        rows = lineage_run("K")
        baseline = rows[0]
        assert baseline["variant"] == "ESCOTT"
        assert baseline["lineage"] == "K" and baseline["lineage_key"] == "K"
        assert baseline["parent_lineage"] is None
        assert baseline["is_primary_parent"] is None      # not False -- no parent at all
        assert baseline["frequency_path"] is None
        assert baseline["equation"] is None and baseline["coefficient"] is None
        assert baseline["n_flat_columns"] == len(C.ESCOTT_FLAT_POSITIONS)
        assert baseline["n_mutants_with_frequency"] == 0
        assert baseline["temperature"] == 1.0
        assert Path(baseline["score_matrix_path"]).exists()
        assert baseline["md5"] == C.md5_file(Path(baseline["score_matrix_path"]))

    def test_the_grid_is_the_cartesian_product(self, lineage_run, fake_tools):
        fake_tools()
        rows = lineage_run("K", coefficients=(0.25, 0.5), equations=(2, 3),
                           frequency_cutoff_ks=(1, 2))
        assert len(rows) == 1 + 2 * 2 * 2
        names = {row["variant"] for row in rows[1:]}
        assert names == {
            f"PRESCOTT_eq{e}_c{c:.2f}_k{k}_parentJ24".replace(".", "p")
            for e in (2, 3) for c in (0.25, 0.5) for k in (1, 2)
        }

    def test_one_score_matrix_per_variant_on_disk(self, lineage_run, fake_tools):
        fake_tools()
        rows = lineage_run("K", coefficients=(0.25, 0.5))
        written = {p.name for p in lineage_run.scores_dir.glob("*_score_matrix.csv")}
        assert written == {f"K_{row['variant']}_score_matrix.csv" for row in rows}
        assert (lineage_run.scores_dir / "K_ESCOTT_raw.tsv").exists()

    def test_the_raw_dump_is_the_unmodified_escott_matrix(self, lineage_run, fake_tools):
        fake_tools()
        lineage_run("K")
        raw = pd.read_csv(lineage_run.scores_dir / "K_ESCOTT_raw.tsv", sep="\t", index_col=0)
        assert raw.shape == (20, C.QUERY_LENGTH)
        assert raw.isna().sum().sum() == C.QUERY_LENGTH

    def test_the_prescott_rows_carry_the_full_provenance(self, lineage_run, fake_tools):
        fake_tools()
        row = lineage_run("K")[1]
        assert row["parent_lineage"] == "J.2.4"
        assert row["is_primary_parent"] is True
        assert Path(row["frequency_path"]).name == "K_parent_frequency.txt"
        assert row["equation"] == 2 and row["coefficient"] == 0.5
        assert row["frequency_cutoff_k"] == 1
        assert row["parent_median_depth"] == C.PARENT_PANEL_MEDIAN_DEPTH
        assert row["frequency_cutoff"] == pytest.approx(-2.0, abs=1e-12)
        assert row["n_mutants_with_frequency"] == len(C.EXPECTED_FREQUENCY_FILE_MIN_COUNT_1)

    def test_score_matrices_are_readable_and_normalised(self, lineage_run, fake_tools):
        fake_tools()
        rows = lineage_run("K")
        for row in rows:
            frame = pd.read_csv(row["score_matrix_path"], index_col=0, header=None)
            assert list(frame.index) == ["sequence"] + list(C.PLM_CACHE_ROW_ORDER)
            values = frame.iloc[1:].astype(float).to_numpy()
            assert np.allclose(values.sum(axis=0), 1.0, atol=1e-9)

    def test_escott_and_prescott_matrices_actually_differ(self, lineage_run, fake_tools):
        fake_tools()
        rows = lineage_run("K", coefficients=(1.0,))
        escott = pd.read_csv(rows[0]["score_matrix_path"], index_col=0,
                             header=None).iloc[1:].astype(float).to_numpy()
        prescott = pd.read_csv(rows[1]["score_matrix_path"], index_col=0,
                               header=None).iloc[1:].astype(float).to_numpy()
        assert not np.allclose(escott, prescott)
        assert rows[0]["md5"] != rows[1]["md5"]

    def test_coefficient_zero_reproduces_the_escott_baseline(self, lineage_run, fake_tools):
        """The ablation property, end to end through the file writer."""
        fake_tools()
        rows = lineage_run("K", coefficients=(0.0,))
        escott = pd.read_csv(rows[0]["score_matrix_path"], index_col=0,
                             header=None).iloc[1:].astype(float).to_numpy()
        prescott = pd.read_csv(rows[1]["score_matrix_path"], index_col=0,
                               header=None).iloc[1:].astype(float).to_numpy()
        assert np.allclose(escott, prescott, atol=1e-12)

    def test_temperature_is_forwarded_into_the_matrices(self, lineage_run, fake_tools):
        fake_tools()
        rows = lineage_run("K", temperature=0.5)
        assert rows[0]["temperature"] == 0.5
        frame = pd.read_csv(rows[0]["score_matrix_path"], index_col=0, header=None)
        values = frame.iloc[1:].astype(float)
        # a flat column stays flat at any temperature ...
        assert np.allclose(values.iloc[:, C.ESCOTT_FLAT_POSITIONS[0] - 1], 0.05, atol=1e-12)
        # ... and a sharpened column is more peaked than at T = 1
        default = pd.read_csv(
            lineage_run("K")[0]["score_matrix_path"], index_col=0, header=None
        ).iloc[1:].astype(float)
        assert values.iloc[:, 0].max() > default.iloc[:, 0].max()

    def test_an_input_only_lineage_gets_the_escott_variant_only(
        self, lineage_run, fake_tools, capsys
    ):
        fake_tools()
        rows = lineage_run("G.1", parent_lineage=None)
        assert [row["variant"] for row in rows] == ["ESCOTT"]
        assert "no parent frequency file" in capsys.readouterr().out

    def test_flat_columns_are_reported_on_stdout(self, lineage_run, fake_tools, capsys):
        fake_tools()
        lineage_run("K")
        assert f"{len(C.ESCOTT_FLAT_POSITIONS)}/{C.QUERY_LENGTH} positions" in \
            capsys.readouterr().out

    def test_heavy_clipping_is_warned_about(self, lineage_run, fake_tools, capsys):
        fake_tools()
        lineage_run("K", coefficients=(1.0,))
        assert "clipped" in capsys.readouterr().out

    def test_the_escott_cache_is_shared_across_grid_points(self, lineage_run, fake_tools):
        recorder = fake_tools()
        lineage_run("K", coefficients=(0.25, 0.5, 1.0))
        assert len(recorder.escott_calls) == 1

    def test_a_rerun_reuses_the_cached_escott_product(self, lineage_run, fake_tools):
        recorder = fake_tools()
        lineage_run("K")
        lineage_run("K")
        assert len(recorder.escott_calls) == 1

    def test_each_lineage_gets_its_own_workdir(self, lineage_run, fake_tools):
        fake_tools()
        lineage_run("K")
        lineage_run("J.2.4")
        assert (lineage_run.escott_root / "K").is_dir()
        assert (lineage_run.escott_root / "J.2.4").is_dir()
        assert (lineage_run.escott_root / "K" / "HAK_normPred_evolCombi.txt").exists()

    def test_a_wrong_protein_is_caught_by_expect_protein(self, lineage_run, fake_tools):
        """The matrix must belong to THIS lineage's query."""
        fake_tools(protein=C.PARENT_PROTEIN)
        with pytest.raises(ValueError, match="does not match the lineage reference"):
            lineage_run("K")


@pytest.mark.integration
class TestProcessLineageParentSensitivity:

    def test_an_alternate_parent_adds_an_independently_named_variant(
        self, lineage_run, fake_tools
    ):
        fake_tools()
        rows = lineage_run("K", alternate_parents=["J.2_int"])
        variants = [row["variant"] for row in rows]
        assert variants == [
            "ESCOTT",
            "PRESCOTT_eq2_c0p50_k1_parentJ24",
            "PRESCOTT_eq2_c0p50_k1_parentJ2int",
        ]
        assert [row["is_primary_parent"] for row in rows] == [None, True, False]
        assert rows[2]["parent_lineage"] == "J.2_int"
        assert Path(rows[2]["frequency_path"]).name == "K_parentJ2int_frequency.txt"

    def test_the_two_edges_produce_different_matrices(self, lineage_run, fake_tools):
        """The two panels differ only in S25W (frequency 0.01, i.e. log10 = -2.0
        exactly), so the cutoff has to sit BELOW -2.0 for the difference to bite --
        at the tree's default Fc of -2.0 that mutant is exactly at the cutoff and
        equation 2 leaves it alone, making the two matrices byte-identical."""
        fake_tools()
        rows = lineage_run("K", alternate_parents=["J.2_int"], coefficients=(1.0,),
                           frequency_cutoff_mode="fixed", frequency_cutoff_fallback=-3.0)
        assert rows[1]["frequency_cutoff"] == -3.0
        assert rows[1]["n_mutants_with_frequency"] == 6
        assert rows[2]["n_mutants_with_frequency"] == 5
        assert rows[1]["md5"] != rows[2]["md5"]

    def test_at_a_cutoff_of_exactly_the_odd_mutants_frequency_they_coincide(
        self, lineage_run, fake_tools
    ):
        """The complement of the test above, pinned so the coincidence is on the
        record rather than mistaken for a broken sensitivity pass."""
        fake_tools()
        rows = lineage_run("K", alternate_parents=["J.2_int"], coefficients=(1.0,))
        assert rows[1]["frequency_cutoff"] == pytest.approx(-2.0)
        assert rows[1]["md5"] == rows[2]["md5"]

    def test_each_edge_gets_its_own_fc_from_its_own_panel_depth(
        self, lineage_run, fake_tools, prepared_inputs_tree
    ):
        """Reusing one Fc would compare two models under two different penalty scales."""
        report = {
            "K": {"median_mapped_depth": 877.0,
                  "frequency_cutoffs": {"1": -2.9430},
                  "frequency_cutoff_mode": "depth_scaled"},
            "K_parentJ2int_frequency": {"median_mapped_depth": 27452.0,
                                        "frequency_cutoffs": {"1": -4.4386},
                                        "frequency_cutoff_mode": "depth_scaled"},
        }
        (Path(prepared_inputs_tree["inputs_dir"]) / "frequency" /
         "frequency_report.json").write_text(json.dumps(report), encoding="utf-8")
        fake_tools()
        rows = lineage_run("K", alternate_parents=["J.2_int"])
        assert rows[1]["frequency_cutoff"] == pytest.approx(-2.9430)
        assert rows[1]["parent_median_depth"] == 877.0
        assert rows[2]["frequency_cutoff"] == pytest.approx(-4.4386)
        assert rows[2]["parent_median_depth"] == 27452.0

    def test_an_alternate_equal_to_the_primary_is_skipped(
        self, lineage_run, fake_tools, capsys
    ):
        fake_tools()
        rows = lineage_run("K", alternate_parents=["J.2.4"])
        assert len(rows) == 2
        assert "equals the primary parent" in capsys.readouterr().out

    def test_an_unprepared_alternate_parent_is_refused_with_the_fix(
        self, lineage_run, fake_tools
    ):
        fake_tools()
        with pytest.raises(FileNotFoundError) as excinfo:
            lineage_run("K", alternate_parents=["G.1"])
        message = str(excinfo.value)
        assert "K_parentG1_frequency.txt" in message
        assert "--sensitivity-parent-map 'K=G.1'" in message
        assert "Available alternates: ['J.2_int']" in message

    def test_the_sensitivity_pass_is_labelled_on_stdout(self, lineage_run, fake_tools, capsys):
        fake_tools()
        lineage_run("K", alternate_parents=["J.2_int"])
        assert "[SENSITIVITY]" in capsys.readouterr().out


@pytest.mark.integration
class TestProcessLineageParityBlock:

    def test_no_prescott_ref_dir_means_no_prescott_run(self, lineage_run, fake_tools):
        recorder = fake_tools()
        lineage_run("K")
        assert recorder.prescott_calls == []

    def test_the_parity_table_lands_in_the_diagnostics_dir(
        self, lineage_run, fake_tools, tmp_path, prepared_inputs_tree, parsed_matrix
    ):
        frequencies = R.load_frequency_file(
            Path(prepared_inputs_tree["inputs_dir"]) / "frequency" / "K_parent_frequency.txt"
        )
        log10_frequency, _ = R.build_log10_frequency_matrix(frequencies, parsed_matrix)
        details = make_details_frame(parsed_matrix, log10_frequency, 0.5, -2.0, 2)
        fake_tools(details_rows=details)
        diagnostics = tmp_path / "out" / "tables" / "diagnostics"
        lineage_run("K", prescott_ref_dir=tmp_path / "ref", diagnostics_dir=diagnostics)
        parity = pd.read_csv(diagnostics / "prescott_parity_check.tsv", sep="\t")
        assert parity["lineage"].unique().tolist() == ["K"]
        assert len(parity) == C.QUERY_LENGTH * 19
        assert parity["abs_delta_prescott"].max() == 0.0

    def test_without_diagnostics_dir_it_falls_back_beside_the_scores(
        self, lineage_run, fake_tools, tmp_path, prepared_inputs_tree, parsed_matrix
    ):
        frequencies = R.load_frequency_file(
            Path(prepared_inputs_tree["inputs_dir"]) / "frequency" / "K_parent_frequency.txt"
        )
        log10_frequency, _ = R.build_log10_frequency_matrix(frequencies, parsed_matrix)
        fake_tools(details_rows=make_details_frame(parsed_matrix, log10_frequency, 0.5, -2.0, 2))
        lineage_run("K", prescott_ref_dir=tmp_path / "ref")
        assert (lineage_run.scores_dir.parent / "tables" / "diagnostics" /
                "prescott_parity_check.tsv").exists()

    def test_a_rerun_merges_rather_than_duplicating(
        self, lineage_run, fake_tools, tmp_path, prepared_inputs_tree, parsed_matrix
    ):
        frequencies = R.load_frequency_file(
            Path(prepared_inputs_tree["inputs_dir"]) / "frequency" / "K_parent_frequency.txt"
        )
        log10_frequency, _ = R.build_log10_frequency_matrix(frequencies, parsed_matrix)
        fake_tools(details_rows=make_details_frame(parsed_matrix, log10_frequency, 0.5, -2.0, 2))
        diagnostics = tmp_path / "diag"
        lineage_run("K", prescott_ref_dir=tmp_path / "ref", diagnostics_dir=diagnostics)
        lineage_run("K", prescott_ref_dir=tmp_path / "ref", diagnostics_dir=diagnostics)
        parity = pd.read_csv(diagnostics / "prescott_parity_check.tsv", sep="\t")
        assert len(parity) == C.QUERY_LENGTH * 19
        assert not parity.duplicated(subset=["lineage", "mutant"]).any()

    def test_the_parity_run_uses_the_primary_edge_only(
        self, lineage_run, fake_tools, tmp_path, prepared_inputs_tree, parsed_matrix
    ):
        frequencies = R.load_frequency_file(
            Path(prepared_inputs_tree["inputs_dir"]) / "frequency" / "K_parent_frequency.txt"
        )
        log10_frequency, _ = R.build_log10_frequency_matrix(frequencies, parsed_matrix)
        recorder = fake_tools(
            details_rows=make_details_frame(parsed_matrix, log10_frequency, 0.5, -2.0, 2)
        )
        lineage_run("K", alternate_parents=["J.2_int"],
                    prescott_ref_dir=tmp_path / "ref", diagnostics_dir=tmp_path / "diag")
        assert len(recorder.prescott_calls) == 1
        cmd = recorder.prescott_calls[0]["cmd"]
        assert cmd[cmd.index("-g") + 1] == "K_parent_frequency.txt"


# =========================================================================== #
# 13. The variants table, lineage discovery and the CLI
# =========================================================================== #

@pytest.mark.unit
class TestWriteVariantsTable:

    ROWS = [
        {"lineage_key": "K", "variant": "ESCOTT", "md5": "aaa"},
        {"lineage_key": "K", "variant": "PRESCOTT_eq2_c0p50_k1_parentJ24", "md5": "bbb"},
    ]

    def test_it_writes_the_contract_stage_d_reads(self, tmp_path):
        out = R.write_variants_table(self.ROWS, tmp_path / "scores")
        assert out.name == "score_variants.tsv"
        frame = pd.read_csv(out, sep="\t")
        assert frame["variant"].tolist() == ["ESCOTT", "PRESCOTT_eq2_c0p50_k1_parentJ24"]

    def test_a_rerun_replaces_rather_than_appends(self, tmp_path):
        R.write_variants_table(self.ROWS, tmp_path / "scores")
        updated = [dict(row, md5="zzz") for row in self.ROWS]
        out = R.write_variants_table(updated, tmp_path / "scores")
        frame = pd.read_csv(out, sep="\t")
        assert len(frame) == 2
        assert frame["md5"].tolist() == ["zzz", "zzz"]

    def test_rows_for_a_different_lineage_are_kept(self, tmp_path):
        R.write_variants_table(self.ROWS, tmp_path / "scores")
        out = R.write_variants_table(
            [{"lineage_key": "J.2.4", "variant": "ESCOTT", "md5": "ccc"}], tmp_path / "scores"
        )
        frame = pd.read_csv(out, sep="\t")
        assert len(frame) == 3
        assert set(frame["lineage_key"]) == {"K", "J.2.4"}

    def test_the_directory_is_created(self, tmp_path):
        out = R.write_variants_table(self.ROWS, tmp_path / "a" / "b")
        assert out.exists()


@pytest.mark.unit
class TestDiscoverLineages:

    def test_only_lineages_with_a_jet_table_are_scorable(self, prepared_inputs_tree, capsys):
        inputs_dir = Path(prepared_inputs_tree["inputs_dir"])
        (inputs_dir / "jet" / "G.1_surrogate_jet.res").unlink()
        found = R.discover_lineages(inputs_dir)
        assert found == ["J.2.4", "J.2_int", "J_int", "K"]
        assert "skipping lineages with no jet table (input-only): ['G.1']" in \
            capsys.readouterr().out

    def test_g1_first_in_sort_order_is_exactly_why_the_filter_exists(
        self, prepared_inputs_tree
    ):
        """``--test-mode`` takes lineages[:1] and 'G.1' sorts first."""
        inputs_dir = Path(prepared_inputs_tree["inputs_dir"])
        (inputs_dir / "jet" / "G.1_surrogate_jet.res").unlink()
        assert sorted(prepared_inputs_tree["lineages"])[0] == "G.1"
        assert R.discover_lineages(inputs_dir)[0] != "G.1"

    def test_a_missing_query_directory_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="run prepare_inputs.py first"):
            R.discover_lineages(tmp_path / "inputs")

    def test_no_jet_table_anywhere_is_reported(self, prepared_inputs_tree):
        inputs_dir = Path(prepared_inputs_tree["inputs_dir"])
        for path in (inputs_dir / "jet").glob("*.res"):
            path.unlink()
        with pytest.raises(FileNotFoundError, match="run jet_surrogate.py first"):
            R.discover_lineages(inputs_dir)


@pytest.mark.unit
class TestParserAndMain:

    def test_parser_defaults(self):
        args = R.build_parser().parse_args(
            ["--inputs-dir", "i", "--escott-workdir", "w", "--scores-dir", "s"]
        )
        assert args.coefficient_grid == "0.25,0.5,1.0"
        assert args.equation_grid == "2"
        assert args.frequency_cutoff_k == "1"
        assert args.frequency_cutoff_mode == "depth_scaled"
        assert args.frequency_cutoff == -4.0
        assert args.escott_temperature == 1.0
        assert args.alphabet == "lw-i.7"
        assert args.max_coil_length == 5
        assert args.diagnostics_dir is None
        assert args.prescott_ref_dir is None
        assert args.sensitivity_parent_map is None
        assert args.no_parent_sensitivity is False
        assert args.escott_pdbfile is None
        assert args.lineage is None

    def test_lineage_is_repeatable(self):
        args = R.build_parser().parse_args(
            ["--inputs-dir", "i", "--escott-workdir", "w", "--scores-dir", "s",
             "--lineage", "K", "--lineage", "J.2.4"]
        )
        assert args.lineage == ["K", "J.2.4"]

    def test_frequency_cutoff_mode_is_constrained(self):
        with pytest.raises(SystemExit):
            R.build_parser().parse_args(
                ["--inputs-dir", "i", "--escott-workdir", "w", "--scores-dir", "s",
                 "--frequency-cutoff-mode", "guess"]
            )

    def _argv(self, tree, tmp_path, *extra):
        return [
            "--inputs-dir", str(tree["inputs_dir"]),
            "--escott-workdir", str(tmp_path / "escott"),
            "--scores-dir", str(tmp_path / "scores"),
            *extra,
        ]

    def test_main_writes_the_variants_table(self, prepared_inputs_tree, tmp_path, fake_tools):
        fake_tools()
        assert R.main(self._argv(prepared_inputs_tree, tmp_path, "--lineage", "K")) == 0
        frame = pd.read_csv(tmp_path / "scores" / "score_variants.tsv", sep="\t")
        assert set(frame["lineage_key"]) == {"K"}
        assert "ESCOTT" in set(frame["variant"])
        # the default 3-point coefficient grid, x2 because the manifest's
        # sensitivity edge is scored unless it is explicitly turned off
        assert len(frame) == 1 + 3 + 3
        # The ESCOTT baseline's None must survive the TSV round trip as an empty
        # cell, NOT as False -- "conditioned on no parent at all" is not the same
        # claim as "conditioned on a non-primary parent".
        column = frame.set_index("variant")["is_primary_parent"]
        assert column.isna().sum() == 1 and pd.isna(column["ESCOTT"])
        assert bool(column["PRESCOTT_eq2_c0p50_k1_parentJ24"]) is True
        assert bool(column["PRESCOTT_eq2_c0p50_k1_parentJ2int"]) is False
        assert sorted(frame.loc[frame["is_primary_parent"].notna(),
                                "parent_lineage"].unique()) == ["J.2.4", "J.2_int"]

    def test_main_picks_up_the_sensitivity_edge_from_the_manifest(
        self, prepared_inputs_tree, tmp_path, fake_tools, capsys
    ):
        fake_tools()
        R.main(self._argv(prepared_inputs_tree, tmp_path, "--lineage", "K",
                          "--coefficient-grid", "0.5"))
        frame = pd.read_csv(tmp_path / "scores" / "score_variants.tsv", sep="\t")
        assert set(frame["variant"]) == {
            "ESCOTT",
            "PRESCOTT_eq2_c0p50_k1_parentJ24",
            "PRESCOTT_eq2_c0p50_k1_parentJ2int",
        }
        assert "alternate (sensitivity) parent edges" in capsys.readouterr().out

    def test_no_parent_sensitivity_suppresses_the_manifest_edge(
        self, prepared_inputs_tree, tmp_path, fake_tools
    ):
        fake_tools()
        R.main(self._argv(prepared_inputs_tree, tmp_path, "--lineage", "K",
                          "--coefficient-grid", "0.5", "--no-parent-sensitivity"))
        frame = pd.read_csv(tmp_path / "scores" / "score_variants.tsv", sep="\t")
        assert not any("parentJ2int" in name for name in frame["variant"])

    def test_the_cli_sensitivity_map_overrides_the_manifest(
        self, prepared_inputs_tree, tmp_path, fake_tools
    ):
        fake_tools()
        R.main(self._argv(prepared_inputs_tree, tmp_path, "--lineage", "K",
                          "--coefficient-grid", "0.5",
                          "--sensitivity-parent-map", "K=J.2_int"))
        frame = pd.read_csv(tmp_path / "scores" / "score_variants.tsv", sep="\t")
        assert "PRESCOTT_eq2_c0p50_k1_parentJ2int" in set(frame["variant"])

    def test_parent_map_override_changes_the_variant_name(
        self, prepared_inputs_tree, tmp_path, fake_tools
    ):
        fake_tools()
        R.main(self._argv(prepared_inputs_tree, tmp_path, "--lineage", "K",
                          "--coefficient-grid", "0.5", "--no-parent-sensitivity",
                          "--parent-map", "K=J.2_int"))
        frame = pd.read_csv(tmp_path / "scores" / "score_variants.tsv", sep="\t")
        assert "PRESCOTT_eq2_c0p50_k1_parentJ2int" in set(frame["variant"])
        assert frame.loc[frame["variant"] != "ESCOTT", "parent_lineage"].tolist() == ["J.2_int"]

    def test_test_mode_restricts_the_grid_and_the_lineages(
        self, prepared_inputs_tree, tmp_path, fake_tools, capsys
    ):
        (Path(prepared_inputs_tree["inputs_dir"]) / "jet" / "G.1_surrogate_jet.res").unlink()
        fake_tools()
        R.main(self._argv(prepared_inputs_tree, tmp_path, "--test-mode",
                          "--coefficient-grid", "0.25,0.5,1.0",
                          "--equation-grid", "2,3", "--frequency-cutoff-k", "1,2"))
        frame = pd.read_csv(tmp_path / "scores" / "score_variants.tsv", sep="\t")
        assert len(frame) == 2                      # ESCOTT + one PRESCOTT grid point
        assert set(frame["lineage_key"]) == {"J.2.4"}
        assert "--test-mode: restricting to lineage" in capsys.readouterr().out

    def test_test_mode_disables_the_sensitivity_pass(
        self, prepared_inputs_tree, tmp_path, fake_tools
    ):
        for label in ("G.1", "J.2.4", "J.2_int", "J_int"):
            (Path(prepared_inputs_tree["inputs_dir"]) / "jet" /
             f"{label}_surrogate_jet.res").unlink()
        fake_tools()
        R.main(self._argv(prepared_inputs_tree, tmp_path, "--test-mode"))
        frame = pd.read_csv(tmp_path / "scores" / "score_variants.tsv", sep="\t")
        assert set(frame["lineage_key"]) == {"K"}
        assert not any("parentJ2int" in name for name in frame["variant"])

    def test_main_without_lineage_scores_every_discovered_lineage(
        self, prepared_inputs_tree, tmp_path, fake_tools
    ):
        (Path(prepared_inputs_tree["inputs_dir"]) / "jet" / "G.1_surrogate_jet.res").unlink()
        fake_tools()
        R.main(self._argv(prepared_inputs_tree, tmp_path, "--coefficient-grid", "0.5",
                          "--no-parent-sensitivity"))
        frame = pd.read_csv(tmp_path / "scores" / "score_variants.tsv", sep="\t")
        assert set(frame["lineage_key"]) == {"J_int", "J.2_int", "J.2.4", "K"}

    def test_escott_bin_and_env_bin_are_forwarded(
        self, prepared_inputs_tree, tmp_path, fake_tools
    ):
        recorder = fake_tools()
        R.main(self._argv(prepared_inputs_tree, tmp_path, "--lineage", "K",
                          "--coefficient-grid", "0.5", "--no-parent-sensitivity",
                          "--escott-bin", "/opt/escott",
                          "--prescott-env-bin", "/opt/envbin"))
        call = recorder.escott_calls[0]
        assert call["cmd"][0] == "/opt/escott"
        assert call["env"]["PATH"].split(":")[0] == "/opt/envbin"


@pytest.mark.cli
@pytest.mark.requires_prescott_python
class TestCommandLineSurface:

    def test_help_exits_zero_and_documents_the_landmines(self, run_module_cli):
        result = run_module_cli("run_escott", ["--help"])
        assert result.returncode == 0
        assert "--sensitivity-parent-map" in result.stdout
        assert "--diagnostics-dir" in result.stdout
        assert "--no-parent-sensitivity" in result.stdout
        assert "sstjetormaxtwocomponent" in result.stdout

    def test_missing_required_arguments_exit_two(self, run_module_cli):
        result = run_module_cli("run_escott", [])
        assert result.returncode == 2
        assert "--inputs-dir" in result.stderr


# =========================================================================== #
# 14. The real binaries (opt-in behind --run-slow)
# =========================================================================== #

@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.requires_escott
@pytest.mark.requires_r
class TestRealEscott:
    """One real ESCOTT run, on the 12 x 72 synthetic alignment (~3 s)."""

    def test_the_real_product_parses_and_the_naming_constraint_is_real(
        self, tmp_path, escott_inputs
    ):
        jet_md5_before = C.md5_file(escott_inputs["jet"])
        product = R.run_escott_for_lineage(
            "K", escott_inputs["msa"], escott_inputs["jet"], escott_inputs["workdir"]
        )
        workdir = escott_inputs["workdir"]

        # escott really does write <prot>.* as bare CWD filenames ...
        assert product == workdir / "HAK_normPred_evolCombi.txt"
        assert (workdir / "HAK.fasta").exists()
        assert (workdir / "HAK_jet.res").exists()
        # ... and stage B's own table, staged under a different name, is untouched.
        assert C.md5_file(escott_inputs["jet"]) == jet_md5_before

        matrix = R.read_escott_matrix(product, expect_protein=C.QUERY_PROTEIN)
        assert matrix.shape == (20, C.QUERY_LENGTH)
        assert matrix.attrs["wt_sequence"] == C.QUERY_PROTEIN
        assert np.nanmax(matrix.to_numpy()) <= 0.0      # -normPred: negative is deleterious

        # pred.R:487 multiplies column i by trace[i]: every zero-trace position,
        # and only those, comes back as an identically-zero column.
        assert R.count_flat_columns(matrix) == len(C.JET_ZERO_TRACE_POSITIONS)
        for position in C.JET_ZERO_TRACE_POSITIONS:
            assert np.allclose(np.nan_to_num(matrix[position].to_numpy()), 0.0)
        probabilities = R.escott_to_probability(matrix)
        for position in C.JET_ZERO_TRACE_POSITIONS:
            assert np.allclose(probabilities[position].to_numpy(), 0.05, atol=1e-12)

    def test_the_real_run_is_cached(self, tmp_path, escott_inputs):
        first = R.run_escott_for_lineage(
            "K", escott_inputs["msa"], escott_inputs["jet"], escott_inputs["workdir"]
        )
        mtime = first.stat().st_mtime_ns
        second = R.run_escott_for_lineage(
            "K", escott_inputs["msa"], escott_inputs["jet"], escott_inputs["workdir"]
        )
        assert second == first
        assert second.stat().st_mtime_ns == mtime


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.requires_escott
@pytest.mark.requires_r
class TestRealPrescott:
    """Our full-precision reimplementation against the published tool's own output."""

    def test_parity_with_the_real_prescott_binary(self, tmp_path, escott_inputs,
                                                  frequency_file_factory):
        escott_txt = R.run_escott_for_lineage(
            "K", escott_inputs["msa"], escott_inputs["jet"], escott_inputs["workdir"]
        )
        query = C.write_fasta(tmp_path / "K_query.fasta", [(C.QUERY_HEADER, C.QUERY_PROTEIN)])
        frequency = frequency_file_factory(name="K_parent_frequency.txt")
        details_path = R.run_prescott_reference(
            "K", escott_txt, query, frequency, tmp_path / "ref",
            coefficient=0.5, frequency_cutoff=-2.0, equation=2,
        )
        matrix = R.read_escott_matrix(escott_txt)
        log10_frequency, report = R.build_log10_frequency_matrix(
            R.load_frequency_file(frequency), matrix
        )
        assert report["n_unmatched"] == 0
        parity = R.prescott_parity_check(
            matrix, log10_frequency, R.read_prescott_details(details_path),
            coefficient=0.5, frequency_cutoff=-2.0, equation=2,
        )
        assert parity.attrs["passed"] is True
        # prescott.py:952 quantises to 2 dp, so the residual is rounding only.
        assert parity.attrs["max_abs_delta_escott"] <= 0.011
        assert parity.attrs["max_abs_delta_prescott"] <= 0.011
        assert len(parity) == C.QUERY_LENGTH * 19

    def test_the_tools_own_output_is_quantised_and_ours_is_not(
        self, tmp_path, escott_inputs, frequency_file_factory
    ):
        escott_txt = R.run_escott_for_lineage(
            "K", escott_inputs["msa"], escott_inputs["jet"], escott_inputs["workdir"]
        )
        query = C.write_fasta(tmp_path / "K_query.fasta", [(C.QUERY_HEADER, C.QUERY_PROTEIN)])
        frequency = frequency_file_factory(name="K_parent_frequency.txt")
        details = R.read_prescott_details(R.run_prescott_reference(
            "K", escott_txt, query, frequency, tmp_path / "ref",
            coefficient=0.5, frequency_cutoff=-2.0,
        ))
        matrix = R.read_escott_matrix(escott_txt)
        # prescott.py:952 writes with '{:6.2f}', so every value it emits is
        # exactly its own two-decimal rounding.
        theirs = details["ESCOTT"].astype(float)
        assert np.array_equal(theirs.to_numpy(), theirs.round(2).to_numpy())
        # ours is not: 1 - rank/N over a 1440-cell matrix cannot land on 2 dp.
        ours = R.escott_rank_scores(matrix, wildtype_fill="global_max").to_numpy()
        assert not np.array_equal(ours, np.round(ours, 2))
        assert (np.abs(ours - np.round(ours, 2)) > 1e-6).mean() > 0.5


# =========================================================================== #
# 15. The remaining branches: no-op paths that must stay no-ops
# =========================================================================== #

@pytest.mark.unit
class TestQuietPaths:
    """Branches whose only observable behaviour is that nothing happens."""

    def test_a_matrix_with_no_flat_column_prints_no_flat_column_line(
        self, tmp_path, prepared_inputs_tree, fake_tools, capsys
    ):
        fake_tools(flat_positions=())
        rows = R.process_lineage(
            "K", prepared_inputs_tree["inputs_dir"], tmp_path / "escott", tmp_path / "scores",
            parent_lineage="J.2.4", coefficients=(0.5,), equations=(2,),
            frequency_cutoff_ks=(1,),
        )
        assert rows[0]["n_flat_columns"] == 0
        assert "all-zero ESCOTT column" not in capsys.readouterr().out

    def test_light_clipping_is_not_warned_about(
        self, tmp_path, prepared_inputs_tree, fake_tools, capsys
    ):
        fake_tools()
        R.process_lineage(
            "K", prepared_inputs_tree["inputs_dir"], tmp_path / "escott", tmp_path / "scores",
            parent_lineage="J.2.4", coefficients=(0.0,), equations=(2,),
            frequency_cutoff_ks=(1,),
        )
        assert "clipped" not in capsys.readouterr().out

    def test_discover_lineages_is_silent_when_nothing_is_skipped(
        self, prepared_inputs_tree, capsys
    ):
        found = R.discover_lineages(prepared_inputs_tree["inputs_dir"])
        assert len(found) == len(prepared_inputs_tree["lineages"])
        assert "skipping lineages" not in capsys.readouterr().out

    def test_inputs_already_in_the_output_dir_are_not_recopied(
        self, tmp_path, parsed_matrix, frequency_file_factory, fake_tools
    ):
        """``shutil.copy2`` onto itself raises SameFileError, so the staging loop
        must skip a source that already resolves to its target."""
        out_dir = R.ensure_dir(tmp_path / "ref")
        escott_txt = Path(parsed_matrix.attrs["source_path"])
        staged_escott = out_dir / escott_txt.name
        shutil.copy2(escott_txt, staged_escott)
        query = C.write_fasta(out_dir / "K_query.fasta", [(C.QUERY_HEADER, C.QUERY_PROTEIN)])
        frequency = C.write_frequency_file(
            out_dir / "K_parent_frequency.txt", C.EXPECTED_FREQUENCY_FILE_MIN_COUNT_1
        )
        fake_tools()
        details = R.run_prescott_reference(
            "K", staged_escott, query, frequency, out_dir,
            coefficient=0.5, frequency_cutoff=-2.0,
        )
        assert details.exists()

    def test_a_report_entry_without_any_depth_key_falls_through(self, tmp_path, capsys):
        inputs = tmp_path / "inputs"
        (inputs / "frequency").mkdir(parents=True)
        (inputs / "frequency" / "frequency_report.json").write_text(
            json.dumps({"K": {"n_mutants": 6}}), encoding="utf-8"
        )
        assert R.resolve_frequency_cutoff("K", 1, inputs, -4.0) == (-4.0, None)
        assert "no parent depth available" in capsys.readouterr().out

    def test_a_meta_table_without_a_depth_column_falls_through(self, tmp_path):
        inputs = tmp_path / "inputs"
        (inputs / "frequency").mkdir(parents=True)
        (inputs / "frequency" / "K_parent_frequency_meta.tsv").write_text(
            "mutant\tcount\nI10K\t3\n", encoding="utf-8"
        )
        assert R.resolve_frequency_cutoff("K", 1, inputs, -4.0) == (-4.0, None)

    def test_an_empty_meta_table_falls_through(self, tmp_path):
        inputs = tmp_path / "inputs"
        (inputs / "frequency").mkdir(parents=True)
        (inputs / "frequency" / "K_parent_frequency_meta.tsv").write_text(
            "mutant\tdepth\n", encoding="utf-8"
        )
        assert R.resolve_frequency_cutoff("K", 1, inputs, -4.0) == (-4.0, None)


@pytest.mark.unit
class TestModuleConstants:
    """The literals other stages depend on, pinned against independent copies."""

    def test_plm_cache_row_order(self):
        assert R.PLM_CACHE_AA_ORDER == C.PLM_CACHE_ROW_ORDER

    def test_escott_row_order_is_lowercase_alphabetical(self):
        assert R.ESCOTT_AA_ORDER == C.ESCOTT_ROW_ORDER
        assert list(R.ESCOTT_AA_ORDER) == sorted(R.ESCOTT_AA_ORDER)
        assert sorted(aa.upper() for aa in R.ESCOTT_AA_ORDER) == sorted(C.PLM_CACHE_ROW_ORDER)

    def test_the_no_frequency_sentinel(self):
        assert R.NO_FREQUENCY_SENTINEL == 999.0

    def test_equation_four_is_excluded(self):
        assert R.SUPPORTED_PRESCOTT_EQUATIONS == (1, 2, 3, 5)

    def test_lineage_tags_come_from_common(self):
        assert R.LINEAGE_TAGS == C.EXPECTED_LINEAGE_TAGS

    def test_the_naming_helpers_are_the_shared_ones(self):
        assert R.parse_edge_spec is constants.parse_edge_spec
        assert R.alternate_frequency_basename is constants.alternate_frequency_basename
        assert R.variant_parent_token is constants.variant_parent_token
        assert R.alternate_frequency_basename("K", "J.2_int") == "K_parentJ2int_frequency"

    def test_a_frequency_file_that_matches_nothing_is_refused(
        self, tmp_path, prepared_inputs_tree, fake_tools, capsys
    ):
        """UPDATED: this used to assert the run CONTINUED with an empty prior.

        It did, and that was the bug.  With no record landing, every PRESCOTT
        equation skips every cell and ``prescott_v2_scores`` collapses to the
        identity (measured 5.6e-17 against the ESCOTT baseline), so stage C
        emitted a PRESCOTT variant that was a numerical clone of ESCOTT and stage
        D reported it as a separate model.  ``assert_frequency_frame`` now refuses.
        See ``test_regressions_coordinates.TestFrequencyFileMustBeInTheEscottFrame``
        for the measurement and for the clipping-denominator case this test used
        to stand in for.
        """
        C.write_frequency_file(
            Path(prepared_inputs_tree["inputs_dir"]) / "frequency" / "K_parent_frequency.txt",
            {"W900A": 0.5, "W901A": 0.5},
        )
        fake_tools()
        with pytest.raises(ValueError, match="not in the ESCOTT column frame"):
            R.process_lineage(
                "K", prepared_inputs_tree["inputs_dir"], tmp_path / "escott", tmp_path / "scores",
                parent_lineage="J.2.4", coefficients=(1.0,), equations=(2,),
                frequency_cutoff_ks=(1,),
            )
        assert "did not match the ESCOTT frame" in capsys.readouterr().out


@pytest.mark.unit
class TestStandaloneFallbacks:
    """The module must still work if ``common.py`` is unavailable.

    Its docstring promises the local definitions are a fallback, not a fork, so
    the two implementations have to agree on every value the pipeline uses.
    """

    @pytest.fixture()
    def without_common(self, monkeypatch):
        monkeypatch.setattr(R, "_common", None)

    @pytest.mark.parametrize("label", ["J.2_int", "J.2.4", "K", "G.1", " A/B "])
    def test_safe_label_agrees(self, label, without_common):
        expected = {"J.2_int": "J.2_int", "J.2.4": "J.2.4", "K": "K",
                    "G.1": "G.1", " A/B ": "A-B"}[label]
        assert R.safe_label(label) == expected

    @pytest.mark.parametrize("label", ["J.2_int", "J.2.4", "K"])
    def test_dotfree_key_agrees(self, label, without_common):
        assert R.dotfree_key(label) == label.replace(".", "_")

    def test_file_md5_agrees(self, tmp_path, without_common):
        path = tmp_path / "x.bin"
        path.write_bytes(b"payload" * 500)
        assert R.file_md5(path) == C.md5_file(path)

    def test_escott_prot_token_agrees(self, tmp_path, without_common):
        path = C.write_fasta(tmp_path / "q.fasta", [("BLAT/1-286", "MKT")])
        assert R.escott_prot_token(path) == "BLAT"
