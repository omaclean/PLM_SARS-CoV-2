#!/usr/bin/env python3
"""The single source of truth for constants shared across BOTH halves of the pipeline.

WHY THIS FILE EXISTS
--------------------
The whole pipeline now runs under ONE conda env (``PRESCOTT``), but it is still
split across a process boundary: stage 1 (``prepare_inputs.py`` /
``jet_surrogate.py`` / ``run_escott.py``) is invoked as a set of subprocesses by
stage 2 (``scripts/run_prescott_diversity.py``), because each stage-1 step is
independently cacheable, resumable and runnable by hand.

That leaves the parent map, the input-only lineage set and the escott name tokens
duplicated on both sides of a process boundary, which is exactly the kind of
duplication that rots: change ``common.PARENT_MAP_PRESETS`` (the map that decides
which panel the frequency prior is built from) and the driver keeps its own copy,
so ``panel_metadata.parent_lineage`` and ``run_manifest.parent_map`` would name a
parent that PRESCOTT never conditioned on -- with no error anywhere.

``run_prescott_diversity.load_prescott_iav_constants`` already looks for
``prescott_iav/constants.py`` and lets it win over the driver's local literals.
This module is that file.  It is deliberately:

* **dependency-free** -- stdlib typing only, so it imports anywhere and can never
  drag prody/scipy/torch into a process that does not want them;
* **import-safe from anywhere** -- as a package member (``prescott_iav.constants``),
  as a bare module on ``sys.path`` (``import constants``), and as a relative import
  from its siblings;
* **the authority** -- ``common.py`` re-exports from here rather than redefining,
  and ``jet_surrogate.py`` takes its tuned defaults from here.

Nothing in this file may import a sibling module: everything else imports *it*.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Tuple

__all__ = [
    "DEFAULT_PARENT_MAPS",
    "DEFAULT_PARENT_MAP_PRESET",
    "INPUT_ONLY_LINEAGES",
    "LINEAGE_TAGS",
    "DEFAULT_TRACE_TOP_FRACTION",
    "MAX_ZERO_TRACE_FRACTION",
    "WARN_ZERO_TRACE_FRACTION",
    "BLAT_REFERENCE_JET_RES",
    "BLAT_REFERENCE_MSA",
    "BLAT_REFERENCE_PDB",
    "dot_free_key",
    "variant_parent_token",
    "alternate_frequency_basename",
    "sensitivity_edges_between_presets",
    "preset_names",
    "parse_edge_spec",
]


# --------------------------------------------------------------------------- #
# Lineage topology
# --------------------------------------------------------------------------- #

# H3N2 clade topology, nextstrain 2024-25 nomenclature:
#   G.1 -> ... -> G.1.3.1 -> J -> J.2 -> J.2.4 -> J.2.4.1 (= K)
#
# `clade_evidence` is the default and is what the on-disk evidence supports: K's
# own FASTA header carries the clade call `J.2.4.1` (Sequences/IAV_lineage_files/
# K.nt.fa, corroborated by Sequences/nextclade_id_clade.tsv), so its basal panel is
# J.2.4.  `brief_as_stated` reproduces the original project brief's K <- J.2_int
# edge so the two can be compared in one run; that disagreement is the single
# contested edge and is what --parent-sensitivity exists to test.
DEFAULT_PARENT_MAPS: Dict[str, Dict[str, str]] = {
    "clade_evidence": {
        "J_int": "G.1",
        "J.2_int": "J_int",
        "J.2.4": "J.2_int",
        "K": "J.2.4",
    },
    "brief_as_stated": {
        "J_int": "G.1",
        "J.2_int": "J_int",
        "J.2.4": "J.2_int",
        "K": "J.2_int",
    },
}

DEFAULT_PARENT_MAP_PRESET = "clade_evidence"

# Lineages with no basal panel underneath them. They are still *prepared* (they are
# somebody's parent, so their panel supplies a frequency prior) but never scored,
# because there is nothing to condition on.
INPUT_ONLY_LINEAGES: FrozenSet[str] = frozenset({"G.1"})

# escott derives its `prot` token by splitting the FASTA header on the first
# non-alphanumeric character (escott.py:76), then names *every* output after it.
# Pinning the tags keeps those filenames short, unique and dot-free.
LINEAGE_TAGS: Dict[str, str] = {
    "G.1": "G1",
    "J_int": "J",
    "J.2_int": "J2",
    "J.2.4": "J24",
    "K": "K",
}


# --------------------------------------------------------------------------- #
# JET2-surrogate defaults
#
# These live here, not in jet_surrogate.py's argparse block, precisely because the
# driver used to hard-code a *different* number and pass it down unconditionally.
# jet_surrogate.py is the authority; this is where its authority is written down so
# both halves can read the same value.
# --------------------------------------------------------------------------- #

# jet_surrogate.py is the AUTHORITY on this number and reads it from here; no
# caller may hard-code a different one.  Re-measured 2026-08-05 with
# --trace-bootstraps 50 --seed 20260805 on both the BLAT reference inputs
# (aliBLAT.fasta 13473 x 286 + blat-af2.pdb) and the real HA production inputs
# (msa_J_int.fasta 6434 x 566 + 6WXB_chainA_qnum.pdb / 6WXB_trimer_qnum.pdb,
# --weight-mode structural, i.e. exactly what stage 1 runs):
#
#     inputs            weight mode   top    zero-trace cols     Spearman(trace, JET2)
#     BLAT              tjet          0.30   196/286  (68.5%)    0.704
#     BLAT              tjet          0.90    28/286  ( 9.8%)    0.877
#     BLAT              structural    0.30    30/286  (10.5%)    0.770
#     BLAT              structural    0.90    23/286  ( 8.0%)    0.837
#     HA J_int          structural    0.30    77/566  (13.6%)    n/a
#     HA J_int          structural    0.90    18/566  ( 3.2%)    n/a
#     BLAT real JET2    --            --       3/286  ( 1.0%)    1.000
#
# 0.90 wins on EVERY measured column (tr 0.877 vs 0.703, freq 0.582 vs 0.478,
# trace 0.877 vs 0.704) and on zero-trace fraction in every configuration, so
# there is no axis on which 0.30 is defensible.  It matters because pred.R:487
# does `normPred[,i] = normPred[,i] * trace[i]`: a zero-trace column comes back
# identically zero, the per-column softmax turns it into a uniform 1/20, and the
# site contributes pure noise to every site-level metric.
DEFAULT_TRACE_TOP_FRACTION = 0.90

# The share of emitted `trace` values allowed to be exactly zero before
# jet_surrogate REFUSES to write the table (see ZeroTraceError).  Chosen from the
# table above: it is the one cut that accepts every top_fraction=0.90 run
# (9.8% / 8.0% / 3.2%) and rejects every top_fraction=0.30 run (68.5% / 10.5% /
# 13.6%), which is precisely the failure this guard exists to make impossible.
# Real JET2 sits at 1.0% on BLAT, so 10% is already ten times more permissive
# than the thing we are imitating.  Override per-run with
# --max-zero-trace-fraction (1.0 disables); the warning below fires regardless.
MAX_ZERO_TRACE_FRACTION = 0.10

# Half the refusal ceiling: above this we warn but still write, because a table
# that is 5-10% dead is usable-but-suspect rather than broken.
WARN_ZERO_TRACE_FRACTION = 0.05

# The only real JET2 output shipped with PRESCOTT, and the inputs it was produced
# from.  `--validate-against` re-runs the surrogate on these and reports the
# per-column agreement, which is the only honest check we can make of a surrogate
# whose training data we do not have.
BLAT_REFERENCE_JET_RES = "/home3/oml4h/PRESCOTT/data/BLAT_jet.res"
BLAT_REFERENCE_MSA = "/home3/oml4h/PRESCOTT/data/aliBLAT.fasta"
BLAT_REFERENCE_PDB = "/home3/oml4h/PRESCOTT/data/blat-af2.pdb"


# --------------------------------------------------------------------------- #
# Naming helpers
#
# These are here rather than in common.py because run_escott.build_variant_name and
# prepare_inputs' alternate-parent frequency filenames must agree *exactly* -- the
# variant token is what stage 2 parses back out of `PRESCOTT_..._parent<TOK>`.
# --------------------------------------------------------------------------- #

def dot_free_key(lineage_key: str) -> str:
    """``J.2_int`` -> ``J_2_int``.

    ``prescott.py:902`` runs ``os.path.splitext`` on its ``-o`` value, which
    truncates at the *last* dot, so nothing with a dot may reach it.
    """
    return lineage_key.replace(".", "_")


def variant_parent_token(parent_lineage: str) -> str:
    """The ``<TOK>`` in ``PRESCOTT_eq2_c0p50_k1_parent<TOK>``.

    ``J.2_int`` -> ``J2int``; ``J.2.4`` -> ``J24``.  Byte-identical to what
    ``run_escott.build_variant_name`` and ``run_prescott_diversity.variant_token``
    produce -- the driver reads the suffix back to decide whether a row is a
    primary model or a parent-sensitivity row, so a one-character drift here
    silently reclassifies every sensitivity variant.
    """
    return dot_free_key(str(parent_lineage)).replace("_", "")


def alternate_frequency_basename(lineage_key: str, parent_lineage: str) -> str:
    """Filename stem for a NON-primary parent's frequency file.

    The primary parent keeps the historical ``<key>_parent_frequency.txt`` (no
    token), so existing trees stay readable; an alternate parent edge gets
    ``<key>_parent<TOK>_frequency.txt``, e.g. ``K_parentJ2int_frequency.txt``.
    The two forms cannot collide: the primary always has an underscore where the
    alternate has its token.
    """
    return f"{lineage_key}_parent{variant_parent_token(parent_lineage)}_frequency"


def sensitivity_edges_between_presets(
    preset: str,
    other_preset: str = "",
) -> Dict[str, str]:
    """Edges on which two presets disagree -- the only ones worth a second pass.

    Everywhere the presets agree, a sensitivity variant would be a byte-identical
    duplicate of the primary.  Mirrors ``run_prescott_diversity.sensitivity_edges``
    so the driver and stage 1 plan the same set.
    """
    if preset not in DEFAULT_PARENT_MAPS:
        raise ValueError(f"Unknown parent-map preset {preset!r}")
    if not other_preset:
        candidates = [name for name in DEFAULT_PARENT_MAPS if name != preset]
        if not candidates:
            return {}
        other_preset = candidates[0]
    primary = DEFAULT_PARENT_MAPS[preset]
    other = DEFAULT_PARENT_MAPS[other_preset]
    return {
        child: other[child]
        for child in primary
        if child in other and other[child] != primary[child]
    }


def preset_names() -> Tuple[str, ...]:
    return tuple(DEFAULT_PARENT_MAPS)


def parse_edge_spec(spec: str) -> Dict[str, str]:
    """``"K=J.2_int,J.2.4=J_int"`` -> ``{"K": "J.2_int", "J.2.4": "J_int"}``.

    Shared by ``prepare_inputs --sensitivity-parent-map`` and
    ``run_escott --sensitivity-parent-map`` so the two stages cannot disagree
    about which alternate edge was requested -- they must, because stage 1 writes
    ``K_parentJ2int_frequency.txt`` and stage 2 has to look for that exact name.
    """
    edges: Dict[str, str] = {}
    for token in str(spec or "").split(","):
        token = token.strip()
        if not token:
            continue
        child, sep, parent = token.partition("=")
        if not sep or not child.strip() or not parent.strip():
            raise ValueError(
                f"malformed edge {token!r}; expected 'child=parent' pairs separated by commas"
            )
        edges[child.strip()] = parent.strip()
    return edges
