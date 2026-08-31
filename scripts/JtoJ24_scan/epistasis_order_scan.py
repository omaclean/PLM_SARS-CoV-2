"""Epistasis scan over every ordering of the mutations separating two lineages.

What it answers
---------------
"How does the protein language model's probability at each mutated site shift
depending on which of the other mutations have already fixed?" — i.e. the
order-dependence (epistasis) of the J -> J.2.4 substitution set.

Method
------
For n mutations there are ``n!`` orderings but only ``2**n`` distinct genotypes,
and the model's output depends only on the genotype, never on the route taken to
reach it. So the scan scores each of the ``2**n`` backgrounds once and
reconstructs all ``n!`` paths from that cache — exact, not an approximation, and
for n = 4 it is 16 forward passes instead of 120.

At every background S and every mutated site i the scan records

    p_wt   = P(wild-type residue at site i | background S)
    p_alt  = P(derived residue at site i   | background S)
    logit  = log(p_alt) - log(p_wt)

and the epistatic shift ``logit(S) - logit(root)``. The log-odds is used as the
headline statistic because it is invariant to the fact that the 20-amino-acid
slice of the model's vocabulary does not sum to 1.

Two scoring schemes are produced by default:

  wt-marginal      the site is left as-is and the model's distribution at that
                   token is read off. Matches the convention already used in
                   ``Notebooks/OM_influenza/Epistasis_hugging_face.py``.
  masked-marginal  the site is replaced with <mask> before the forward pass, so
                   the model cannot see the residue it is being asked to score.
                   The standard, less self-confirming estimator.

Example
-------
    python scripts/JtoJ24_scan/epistasis_order_scan.py \
        --fasta Sequences/huH3N2_HA_CDS.translated.fas \
        --start-id J --end-id J.2.4 \
        --output-dir Results/JtoJ.2.4_scan/epistasis

Run ``--dry-run`` first to check the mutation set and hypercube size without
loading a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # must precede any pyplot import, including transitive ones

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from order_scan_common import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    REPO_ROOT,
    Mutation,
    add_common_arguments,
    all_orderings,
    all_subsets,
    apply_mutations,
    build_h3_map,
    describe_mutation_set,
    genotype_id,
    hypercube_path_extremes,
    order_id,
    path_genotypes,
    resolve_mutation_set,
    subset_to_mask,
    write_fasta,
)

# Row order of the probability matrices, matching Functions_HuggingFace.
AMINO_ACIDS = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
               "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

# ESM2-HA80's config.json reports 33 layers / hidden 1280 / _name_or_path
# facebook/esm2_t33_650M_UR50D, so 650M is the architecture it was fine-tuned
# from. Epistasis_hugging_face.py pairs it with the 3B alphabet instead; that
# works only because every ESM-2 size shares one alphabet, and it costs ~12GB of
# host RAM for a model whose weights are then discarded.
DEFAULT_BASE_MODEL = "esm2_t33_650M_UR50D"
DEFAULT_CHECKPOINT = Path("/home3/oml4h/hugging_face_downloads/model_weights_topublish/ESM2-HA80")

SCORING_SCHEMES = ("wt-marginal", "masked-marginal")


###############################################################################
# CLI
###############################################################################
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_arguments(parser)

    model = parser.add_argument_group("model")
    model.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="fair-esm architecture name supplying the alphabet/tokeniser (default: %(default)s)",
    )
    model.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Fine-tuned checkpoint directory; pass 'none' to score with the stock "
             "--base-model weights (default: %(default)s)",
    )
    model.add_argument(
        "--model-tag",
        default=None,
        help="Short label used in output filenames (default: checkpoint directory name).",
    )
    model.add_argument(
        "--model-layer",
        type=int,
        default=None,
        help="Hidden-state index to request. Probabilities come from the LM head and are "
             "unaffected; this only matters if you later reuse the embeddings.",
    )
    model.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Sequences per forward pass (default: %(default)s)",
    )
    model.add_argument(
        "--scoring",
        choices=("wt-marginal", "masked-marginal", "both"),
        default="both",
        help="Which probability estimator(s) to compute (default: %(default)s)",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "epistasis",
        help="Directory for all CSVs and figures (default: %(default)s)",
    )
    output.add_argument(
        "--save-matrices",
        action="store_true",
        help="Also write the full 20 x L probability matrix for every genotype.",
    )
    output.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip figure generation; write CSVs only.",
    )
    output.add_argument(
        "--delta-odds-scale",
        choices=("auto", "linear", "symlog"),
        default="auto",
        help="Colour scale for pairwise_delta_odds_*.png. 'auto' switches to symlog when the "
             "largest delta exceeds 50x the median, which happens once a site's WT baseline "
             "odds exceed 1 and its deltas dwarf every other row (default: %(default)s)",
    )
    output.add_argument(
        "--no-all-sites-table",
        action="store_true",
        help="Skip order_paths_all_sites.csv. It has (orderings x steps x sites x scorings) rows, "
             "so it gets large fast; it is a join of order_paths_steps.csv onto "
             "site_probabilities_by_background.csv.",
    )
    output.add_argument(
        "--dry-run",
        action="store_true",
        help="Enumerate mutations, genotypes and orderings, write those tables, then stop "
             "without loading a model.",
    )
    return parser.parse_args(argv)


###############################################################################
# Model runtime
###############################################################################
def load_runtime(base_model: str, checkpoint_dir: Optional[Path]):
    """Load a PLM through the repo's shared loader and return its pieces."""
    sys.path.insert(0, str(REPO_ROOT))
    from Functions_HuggingFace import _load_plm_runtime, resolve_last_layer  # noqa: WPS433

    checkpoint = str(checkpoint_dir) if checkpoint_dir else None
    model, device, batch_converter, alphabet = _load_plm_runtime(base_model, checkpoint)
    return model, device, batch_converter, alphabet, resolve_last_layer(model)


def _tokenize(sequences: Sequence[str], model, batch_converter):
    """Tokenise a batch of equal-length sequences, dispatching on model family."""
    if model.__class__.__name__ == "ESMC":
        return model._tokenize(list(sequences))
    _, _, tokens = batch_converter([(f"s{i}", seq) for i, seq in enumerate(sequences)])
    return tokens


def _forward_log_probs(tokens, model, device):
    """Log-softmax LM-head output for a pre-tokenised batch, returned on CPU.

    Mirrors the dispatch in ``Functions_HuggingFace.embed_sequence`` but asks for
    logits only. Skipping the hidden-state request keeps peak memory down on the
    3B checkpoints; the probabilities are identical to those from
    ``get_mutation_prob_matrix``, which softmaxes the same logits.
    """
    import torch

    tokens = tokens.to(device)
    with torch.no_grad():
        if model.__class__.__name__ == "ESMC":
            results = model.forward(sequence_tokens=tokens)
            logits = results.sequence_logits
        else:
            forward_params = set()
            forward_fn = getattr(model, "forward", None)
            if forward_fn is not None:
                try:
                    forward_params = set(forward_fn.__code__.co_varnames)
                except AttributeError:
                    forward_params = set()

            if "repr_layers" in forward_params:  # fair-esm
                results = model(tokens, repr_layers=[])
                logits = results["logits"]
            else:  # HuggingFace EsmForMaskedLM
                results = model(tokens)
                logits = results.logits if hasattr(results, "logits") else results["logits"]

    return torch.log_softmax(logits.float().cpu(), dim=-1)


def wt_marginal_matrices(
    sequences: Sequence[str],
    model,
    device,
    batch_converter,
    alphabet,
    batch_size: int,
) -> List[np.ndarray]:
    """One 20 x L probability matrix per input sequence, sequence left unmasked."""
    aa_indices = [alphabet.get_idx(aa) for aa in AMINO_ACIDS]
    matrices: List[np.ndarray] = []

    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start : start + batch_size]
        log_probs = _forward_log_probs(_tokenize(chunk, model, batch_converter), model, device)
        for row, sequence in enumerate(chunk):
            # Token 0 is <cls>, so 1-based sequence position p sits at token index p.
            residue_log_probs = log_probs[row, 1 : len(sequence) + 1, :]
            matrices.append(np.exp(residue_log_probs[:, aa_indices].numpy()).T)
    return matrices


def masked_site_probabilities(
    sequence: str,
    positions: Sequence[int],
    model,
    device,
    batch_converter,
    alphabet,
    batch_size: int,
) -> Dict[int, np.ndarray]:
    """P(aa | sequence with `pos` masked) for each requested 1-based position."""
    import torch

    aa_indices = [alphabet.get_idx(aa) for aa in AMINO_ACIDS]
    mask_idx = getattr(alphabet, "mask_idx", None)
    if mask_idx is None:
        mask_idx = alphabet.get_idx("<mask>")

    base_tokens = _tokenize([sequence], model, batch_converter)
    results: Dict[int, np.ndarray] = {}

    for start in range(0, len(positions), batch_size):
        chunk = list(positions[start : start + batch_size])
        tokens = base_tokens.repeat(len(chunk), 1).clone()
        for row, position in enumerate(chunk):
            tokens[row, position] = mask_idx

        log_probs = _forward_log_probs(tokens, model, device)
        for row, position in enumerate(chunk):
            results[position] = np.exp(log_probs[row, position, aa_indices].numpy())

    del base_tokens
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


###############################################################################
# Table assembly
###############################################################################
def build_site_table(
    mutations: Sequence[Mutation],
    subsets: Sequence[Tuple[Mutation, ...]],
    probabilities: Dict[Tuple[str, str], Dict[int, np.ndarray]],
    h3_map: Dict[int, str],
) -> pd.DataFrame:
    """Long table: one row per (scoring, background genotype, mutated site).

    ``probabilities`` is keyed ``(scoring, genotype_id) -> {position: 20-vector}``.
    """
    aa_index = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    rows = []
    scorings = sorted({scoring for scoring, _ in probabilities})
    root_logit: Dict[Tuple[str, int], float] = {}

    for scoring in scorings:
        root_vector = probabilities[(scoring, "root")]
        for mutation in mutations:
            vector = root_vector[mutation.pos]
            root_logit[(scoring, mutation.pos)] = float(
                np.log(vector[aa_index[mutation.alt]]) - np.log(vector[aa_index[mutation.wt]])
            )

    for scoring in scorings:
        for subset in subsets:
            gid = genotype_id(subset)
            fixed = {m.pos for m in subset}
            vectors = probabilities[(scoring, gid)]

            for mutation in mutations:
                vector = vectors[mutation.pos]
                p_wt = float(vector[aa_index[mutation.wt]])
                p_alt = float(vector[aa_index[mutation.alt]])
                logit = float(np.log(p_alt) - np.log(p_wt))
                rows.append(
                    {
                        "scoring": scoring,
                        "background_id": gid,
                        "background_h3": genotype_id(subset, h3_map),
                        "n_fixed": len(subset),
                        "site_pos": mutation.pos,
                        "site_h3": h3_map.get(mutation.pos, str(mutation.pos)),
                        "mutation": mutation.name,
                        "mutation_h3": mutation.label(h3_map),
                        "wt_aa": mutation.wt,
                        "alt_aa": mutation.alt,
                        "is_fixed": mutation.pos in fixed,
                        "p_wt": p_wt,
                        "p_alt": p_alt,
                        "log10_p_alt": float(np.log10(p_alt)),
                        "logit_alt_over_wt": logit,
                        "delta_logit_vs_root": logit - root_logit[(scoring, mutation.pos)],
                        "log2_fold_p_alt_vs_root": float(
                            np.log2(p_alt / probabilities[(scoring, "root")][mutation.pos][aa_index[mutation.alt]])
                        ),
                    }
                )

    return pd.DataFrame(rows)


def build_path_tables(
    orderings: Sequence[Tuple[Mutation, ...]],
    site_table: pd.DataFrame,
    h3_map: Dict[int, str],
    include_all_sites: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-step, per-step-all-sites, and per-path summary tables for every ordering.

    Lookups go through plain dicts rather than pandas indexing: at n = 11 with a
    few thousand sampled orderings this runs hundreds of thousands of times, and
    repeated `.loc` on a MultiIndex (or worse, boolean masking of the whole
    frame) turns a seconds-long assembly into a multi-hour one.
    """
    scorings = sorted(site_table["scoring"].unique())
    columns = ["p_alt", "p_wt", "log10_p_alt", "logit_alt_over_wt", "delta_logit_vs_root"]
    by_site = {
        (row.scoring, row.background_id, row.site_pos): tuple(
            float(getattr(row, column)) for column in columns
        )
        for row in site_table.itertuples(index=False)
    }
    by_background: Dict[Tuple[str, str], List[dict]] = {}
    for row in site_table.itertuples(index=False):
        by_background.setdefault((row.scoring, row.background_id), []).append(
            {
                "site_h3": row.site_h3,
                "mutation_h3": row.mutation_h3,
                "site_pos": int(row.site_pos),
                "is_fixed": bool(row.is_fixed),
                "p_alt": float(row.p_alt),
                "logit_alt_over_wt": float(row.logit_alt_over_wt),
                "delta_logit_vs_root": float(row.delta_logit_vs_root),
            }
        )

    step_rows = []
    site_rows = []
    width = max(3, len(str(len(orderings))))

    for path_index, ordering in enumerate(orderings):
        pid = f"path{path_index + 1:0{width}d}"
        oid = order_id(ordering)
        oid_h3 = order_id(ordering, h3_map)
        nodes = path_genotypes(ordering)
        node_ids = [genotype_id(node) for node in nodes]
        node_h3 = [genotype_id(node, h3_map) for node in nodes]

        for scoring in scorings:
            for step, focal in enumerate(ordering, start=1):
                bid = node_ids[step - 1]
                p_alt, p_wt, log10_p_alt, logit, delta = by_site[(scoring, bid, focal.pos)]

                step_rows.append(
                    {
                        "path_id": pid,
                        "order": oid,
                        "order_h3": oid_h3,
                        "scoring": scoring,
                        "step": step,
                        "background_id": bid,
                        "background_h3": node_h3[step - 1],
                        "focal_mutation": focal.name,
                        "focal_mutation_h3": focal.label(h3_map),
                        "p_alt": p_alt,
                        "p_wt": p_wt,
                        "log10_p_alt": log10_p_alt,
                        "logit_alt_over_wt": logit,
                        "delta_logit_vs_root": delta,
                    }
                )

                if not include_all_sites:
                    continue

                # Every mutated site's standing on this step's background, so a
                # single table shows how the not-yet-fixed sites are being pulled
                # around by the ones that already fixed.
                for record in by_background[(scoring, bid)]:
                    site_rows.append(
                        {
                            "path_id": pid,
                            "order_h3": oid_h3,
                            "scoring": scoring,
                            "step": step,
                            "background_h3": node_h3[step - 1],
                            "focal_mutation_h3": focal.label(h3_map),
                            "site_h3": record["site_h3"],
                            "mutation_h3": record["mutation_h3"],
                            "is_fixed": record["is_fixed"],
                            "is_focal": record["site_pos"] == focal.pos,
                            "p_alt": record["p_alt"],
                            "logit_alt_over_wt": record["logit_alt_over_wt"],
                            "delta_logit_vs_root": record["delta_logit_vs_root"],
                        }
                    )

    steps = pd.DataFrame(step_rows)
    sites = pd.DataFrame(site_rows)

    summaries = []
    for (pid, scoring), group in steps.groupby(["path_id", "scoring"], sort=False):
        bottleneck = group.loc[group["log10_p_alt"].idxmin()]
        summaries.append(
            {
                "path_id": pid,
                "scoring": scoring,
                "order": group["order"].iloc[0],
                "order_h3": group["order_h3"].iloc[0],
                "sum_log10_p_alt": float(group["log10_p_alt"].sum()),
                "sum_logit": float(group["logit_alt_over_wt"].sum()),
                "mean_delta_logit_vs_root": float(group["delta_logit_vs_root"].mean()),
                "bottleneck_step": int(bottleneck["step"]),
                "bottleneck_mutation_h3": bottleneck["focal_mutation_h3"],
                "bottleneck_log10_p_alt": float(bottleneck["log10_p_alt"]),
            }
        )

    summary = pd.DataFrame(summaries)
    summary["rank_by_sum_log10_p_alt"] = summary.groupby("scoring")["sum_log10_p_alt"].rank(
        ascending=False, method="min"
    ).astype(int)
    summary = summary.sort_values(["scoring", "rank_by_sum_log10_p_alt"]).reset_index(drop=True)
    return steps, sites, summary


def build_exact_extremes(
    mutations: Sequence[Mutation],
    site_table: pd.DataFrame,
    h3_map: Dict[int, str],
) -> pd.DataFrame:
    """Best- and worst-scoring ordering over *all* n!, via the hypercube DP.

    The sampled `order_paths_summary.csv` ranking only covers the orderings that
    were drawn. This covers every one of them exactly, which is the difference
    between "best of 2000 sampled" and "best of 39,916,800" once n gets past 8.
    """
    ordered = sorted(mutations)
    index_of = {m.pos: i for i, m in enumerate(ordered)}
    mask_of = {genotype_id(subset): subset_to_mask(ordered, subset)
               for subset in all_subsets(ordered)}

    rows = []
    for scoring in sorted(site_table["scoring"].unique()):
        subset_table = site_table[site_table["scoring"] == scoring]
        for metric, column in (("sum_log10_p_alt", "log10_p_alt"),
                               ("sum_logit", "logit_alt_over_wt")):
            scores = {
                (mask_of[row.background_id], index_of[row.site_pos]): float(getattr(row, column))
                for row in subset_table.itertuples(index=False)
                if not row.is_fixed
            }
            result = hypercube_path_extremes(ordered, lambda mask, i: scores[(mask, i)])
            for kind in ("best", "worst"):
                rows.append(
                    {
                        "scoring": scoring,
                        "metric": metric,
                        "extreme": kind,
                        "order_h3": order_id(result[f"{kind}_ordering"], h3_map),
                        "order": order_id(result[f"{kind}_ordering"]),
                        "score": result[f"{kind}_score"],
                        "n_orderings_covered": result["n_orderings_covered"],
                    }
                )
    return pd.DataFrame(rows)


def build_pairwise_epistasis(
    mutations: Sequence[Mutation],
    site_table: pd.DataFrame,
    h3_map: Dict[int, str],
) -> pd.DataFrame:
    """Double-mutant-cycle epistasis on the log-odds scale.

    ``eps_focal_given_context = logit(focal | {context}) - logit(focal | root)``.
    A symmetric energy function would give the same number with focal and context
    swapped; a PLM need not, so both directions and their difference are kept.
    """
    lookup = site_table.set_index(["scoring", "background_id", "site_pos"])["logit_alt_over_wt"]
    rows = []

    for scoring in sorted(site_table["scoring"].unique()):
        for focal in mutations:
            for context in mutations:
                if focal.pos == context.pos:
                    continue
                alone = float(lookup.loc[(scoring, "root", focal.pos)])
                conditioned = float(lookup.loc[(scoring, genotype_id((context,)), focal.pos)])
                reverse_alone = float(lookup.loc[(scoring, "root", context.pos)])
                reverse_conditioned = float(
                    lookup.loc[(scoring, genotype_id((focal,)), context.pos)]
                )
                rows.append(
                    {
                        "scoring": scoring,
                        "focal_h3": focal.label(h3_map),
                        "context_h3": context.label(h3_map),
                        "logit_focal_alone": alone,
                        "logit_focal_given_context": conditioned,
                        "eps_focal_given_context": conditioned - alone,
                        "eps_reverse": reverse_conditioned - reverse_alone,
                        "asymmetry": (conditioned - alone) - (reverse_conditioned - reverse_alone),
                    }
                )
    return pd.DataFrame(rows)


def build_offtarget_shifts(
    subsets: Sequence[Tuple[Mutation, ...]],
    matrices: Dict[str, np.ndarray],
    h3_map: Dict[int, str],
    root_sequence: str,
) -> pd.DataFrame:
    """Total absolute probability shift per position, each genotype vs the root.

    Summed over the 20 amino acids, so the ceiling per site is 2.0. This is the
    off-target half of the picture: epistasis the mutations exert on sites that
    are not themselves mutated.
    """
    root = matrices["root"]
    frame = pd.DataFrame(
        {
            "position": np.arange(1, root.shape[1] + 1),
            "h3_label": [h3_map.get(p, str(p)) for p in range(1, root.shape[1] + 1)],
            "root_aa": list(root_sequence[: root.shape[1]]),
        }
    )
    for subset in subsets:
        gid = genotype_id(subset)
        if gid == "root":
            continue
        frame[gid] = np.nansum(np.abs(matrices[gid] - root), axis=0)
    return frame


###############################################################################
# Figures
###############################################################################
def plot_delta_logit_heatmap(site_table: pd.DataFrame, scoring: str, path: Path) -> None:
    subset = site_table[site_table["scoring"] == scoring].copy()
    # A site that has already fixed cannot "acquire" its mutation, so blank it out
    # rather than plotting a number that means something different from its neighbours.
    subset.loc[subset["is_fixed"], "delta_logit_vs_root"] = np.nan
    matrix = subset.pivot_table(
        index="mutation_h3", columns="background_h3", values="delta_logit_vs_root"
    )
    order = sorted(matrix.columns, key=lambda c: (0 if c == "root" else c.count("+") + 1, c))
    matrix = matrix[order]

    height = max(3.0, 0.7 * len(matrix.index) + 2.0)
    width = max(8.0, 0.75 * len(matrix.columns) + 4.0)
    plt.figure(figsize=(width, height))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="vlag", center=0,
                cbar_kws={"label": "Δ log-odds vs J background"})
    plt.title(f"Epistatic shift at each mutated site by background ({scoring})")
    plt.xlabel("Background genotype (mutations already fixed)")
    plt.ylabel("Focal mutation")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_pairwise_heatmap(pairwise: pd.DataFrame, scoring: str, path: Path) -> None:
    subset = pairwise[pairwise["scoring"] == scoring]
    matrix = subset.pivot_table(
        index="focal_h3", columns="context_h3", values="eps_focal_given_context"
    )
    plt.figure(figsize=(max(6.0, 1.1 * len(matrix.columns) + 3), max(5.0, 1.0 * len(matrix.index) + 2)))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="vlag", center=0,
                cbar_kws={"label": "Δ log-odds of focal given context"})
    plt.title(f"Pairwise epistasis ({scoring})")
    plt.xlabel("Context mutation (already fixed)")
    plt.ylabel("Focal mutation")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def build_odds_with_wt(site_table: pd.DataFrame, scoring: str) -> pd.DataFrame:
    """Raw odds P(derived)/P(ancestral) for each focal mutation on each single-mutant background.

    Columns are ``WT`` (the unmutated start background) followed by one column per
    other mutation. Values are **raw odds**, not a shift: the WT column is the
    baseline the Δ-log-odds view measures against, so on that view it would be a
    column of zeros and carry no information. Here it carries the thing every
    other column should be read against — how much the model liked the mutation
    before any of its partners arrived.
    """
    subset = site_table[(site_table["scoring"] == scoring) & (site_table["n_fixed"] <= 1)]
    focal_order = (
        subset[["site_pos", "mutation_h3"]]
        .drop_duplicates()
        .sort_values("site_pos")["mutation_h3"]
        .tolist()
    )

    odds = np.exp(subset["logit_alt_over_wt"])
    matrix = (
        subset.assign(odds=odds)
        .pivot_table(index="mutation_h3", columns="background_h3", values="odds")
        .reindex(index=focal_order)
    )

    # The single-mutant background whose own site is the focal one is the mutation
    # already fixed, not a context for acquiring it, so it is not comparable.
    for label in matrix.index:
        if label in matrix.columns:
            matrix.loc[label, label] = np.nan

    columns = ["root"] + [c for c in focal_order if c in matrix.columns]
    matrix = matrix[[c for c in columns if c in matrix.columns]]
    return matrix.rename(columns={"root": "WT"})


def plot_odds_with_wt(
    site_table: pd.DataFrame,
    scoring: str,
    figure_path: Path,
    csv_path: Optional[Path] = None,
) -> None:
    """Heatmap of raw odds by single-mutant context, with the WT baseline column."""
    from matplotlib.colors import LogNorm

    matrix = build_odds_with_wt(site_table, scoring)
    if csv_path is not None:
        matrix.to_csv(csv_path)

    values = matrix.values[np.isfinite(matrix.values)]
    if values.size == 0:
        return
    norm = LogNorm(vmin=float(values.min()), vmax=float(values.max()))

    width = max(7.0, 1.15 * len(matrix.columns) + 3.5)
    height = max(4.5, 0.62 * len(matrix.index) + 2.5)
    plt.figure(figsize=(width, height))
    axis = sns.heatmap(
        matrix, annot=True, fmt=".3g", cmap="viridis", norm=norm,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "odds P(derived) / P(ancestral), log scale"},
        annot_kws={"size": 8},
    )
    # Set the WT baseline apart: it is a different kind of column from the rest.
    if "WT" in matrix.columns:
        axis.axvline(1.0, color="black", linewidth=2.0)

    plt.title(
        f"Odds of each mutation by single-mutant context ({scoring})\n"
        "column WT = unmutated start background; blank diagonal = mutation is the context"
    )
    plt.xlabel("Background (mutation already fixed)")
    plt.ylabel("Focal mutation")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()


def build_delta_odds_vs_wt(
    site_table: pd.DataFrame,
    scoring: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Change in odds relative to the WT background, per focal mutation and context.

    ``odds(focal | context) - odds(focal | WT)``. Returns the delta matrix and the
    WT baseline it was measured against.

    The raw-odds view is dominated by *which* mutation a row is — N122D's odds sit
    an order of magnitude above K276E's, so its whole row saturates the colour
    scale and the context effect within every row is invisible. Subtracting the WT
    baseline removes that per-mutation offset, leaving only what the context did.
    The WT column is dropped rather than plotted as a column of zeros; its value is
    folded into the row labels instead, so the baseline each delta is measured
    against is still on the figure.
    """
    odds = build_odds_with_wt(site_table, scoring)
    if "WT" not in odds.columns:
        raise ValueError("No WT (root) background in the site table; cannot take a delta.")
    baseline = odds["WT"]
    delta = odds.drop(columns=["WT"]).sub(baseline, axis=0)
    return delta, baseline


def plot_delta_odds_vs_wt(
    site_table: pd.DataFrame,
    scoring: str,
    figure_path: Path,
    csv_path: Optional[Path] = None,
    scale: str = "auto",
) -> None:
    """Heatmap of Δ odds vs the WT background, on a diverging scale centred at zero.

    `scale` controls only how values map to colour, never what is plotted:

    linear  equal odds-units get equal colour distance.
    symlog  linear near zero, logarithmic beyond. Needed when WT baseline odds
            span orders of magnitude -- a site the model already prefers the
            derived residue at (odds >> 1) produces deltas thousands of times
            larger than a site it does not, and on a linear scale that one row
            whites out every other.
    auto    symlog when max|Δ| exceeds 50x the median |Δ|, else linear.
    """
    from matplotlib.colors import SymLogNorm

    delta, baseline = build_delta_odds_vs_wt(site_table, scoring)
    if csv_path is not None:
        delta.assign(WT_baseline_odds=baseline).to_csv(csv_path)

    values = delta.values[np.isfinite(delta.values)]
    if values.size == 0:
        return
    # Symmetric limits so zero sits exactly at the neutral midpoint of the
    # diverging map and equal gains and losses read as equally intense.
    magnitudes = np.abs(values)
    limit = float(magnitudes.max())
    nonzero = magnitudes[magnitudes > 0]
    spread = limit / float(np.median(nonzero)) if nonzero.size else 1.0

    if scale == "auto":
        scale = "symlog" if spread > 50 else "linear"
    if scale == "symlog" and nonzero.size:
        norm = SymLogNorm(linthresh=float(np.median(nonzero)), vmin=-limit, vmax=limit, base=10)
    else:
        scale = "linear"
        norm = plt.Normalize(vmin=-limit, vmax=limit)

    labelled = delta.copy()
    labelled.index = [f"{label}   (WT {baseline[label]:.3g})" for label in delta.index]

    width = max(7.0, 1.3 * len(labelled.columns) + 4.5)
    height = max(4.5, 0.62 * len(labelled.index) + 2.5)
    plt.figure(figsize=(width, height))
    sns.heatmap(
        labelled, annot=True, fmt="+.3g", cmap="vlag", norm=norm,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Δ odds vs WT background"},
        annot_kws={"size": 8},
    )
    plt.title(
        f"Change in odds vs the WT background ({scoring}, {scale} colour scale)\n"
        "odds(focal | context) − odds(focal | WT); blank diagonal = mutation is the context"
    )
    plt.xlabel("Background (mutation already fixed)")
    plt.ylabel("Focal mutation (WT baseline odds in brackets)")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()


def plot_path_ranking(steps: pd.DataFrame, summary: pd.DataFrame, scoring: str, path: Path) -> None:
    ranked = summary[summary["scoring"] == scoring].sort_values("sum_log10_p_alt", ascending=False)
    step_subset = steps[steps["scoring"] == scoring]

    fig, axes = plt.subplots(
        1, 2, figsize=(16, max(5.0, 0.32 * len(ranked) + 3)), sharey=True,
        gridspec_kw={"width_ratios": [1.4, 1.0]},
    )

    labels = ranked["order_h3"].tolist()
    y = np.arange(len(labels))[::-1]

    axes[0].barh(y, ranked["sum_log10_p_alt"], color="#4c72b0")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlabel("Σ log10 P(derived residue) along the path")
    axes[0].set_title(f"Path accessibility ranking ({scoring})")
    axes[0].grid(axis="x", alpha=0.3)

    palette = sns.color_palette("viridis", n_colors=int(step_subset["step"].max()))
    for offset, (_, row) in enumerate(ranked.iterrows()):
        path_steps = step_subset[step_subset["path_id"] == row["path_id"]].sort_values("step")
        axes[1].scatter(
            path_steps["log10_p_alt"],
            np.full(len(path_steps), y[offset]),
            c=[palette[int(s) - 1] for s in path_steps["step"]],
            s=42,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
    axes[1].set_xlabel("log10 P(derived residue) at each step")
    axes[1].set_title("Per-step probabilities (colour = step index)")
    axes[1].grid(axis="x", alpha=0.3)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=palette[i], label=f"step {i + 1}")
        for i in range(len(palette))
    ]
    axes[1].legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_site_context_spread(site_table: pd.DataFrame, scoring: str, path: Path) -> None:
    subset = site_table[(site_table["scoring"] == scoring) & (~site_table["is_fixed"])]
    plt.figure(figsize=(9, 6))
    sns.stripplot(data=subset, x="mutation_h3", y="logit_alt_over_wt", hue="n_fixed",
                  palette="viridis", size=9, jitter=0.18, dodge=False)
    plt.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    plt.xlabel("Mutation")
    plt.ylabel("log-odds derived vs wild-type residue")
    plt.title(f"Context dependence of each mutation ({scoring})\n"
              "one point per background genotype in which the site is still ancestral")
    plt.legend(title="mutations already fixed", fontsize=8, title_fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_offtarget_panel(
    offtarget: pd.DataFrame,
    mutations: Sequence[Mutation],
    path: Path,
) -> None:
    genotype_columns = [c for c in offtarget.columns if c not in {"position", "h3_label", "root_aa"}]
    columns = int(np.ceil(np.sqrt(len(genotype_columns))))
    rows = int(np.ceil(len(genotype_columns) / columns))

    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 2.8 * rows),
                             sharex=True, sharey=True)
    flat = np.atleast_1d(axes).ravel()
    focal_positions = [m.pos for m in mutations]

    masked = offtarget.copy()
    for column in genotype_columns:
        masked.loc[masked["position"].isin(focal_positions), column] = np.nan
    ceiling = float(np.nanmax(masked[genotype_columns].values)) if genotype_columns else 1.0

    for index, column in enumerate(genotype_columns):
        axis = flat[index]
        axis.plot(masked["position"], masked[column], linewidth=0.8, color="#2b5d8c")
        for position in focal_positions:
            axis.axvline(position, color="orange", alpha=0.45, linewidth=1.4)
        axis.set_title(column, fontsize=8)
        axis.set_ylim(0, ceiling * 1.1)
    for index in range(len(genotype_columns), len(flat)):
        flat[index].axis("off")

    fig.suptitle(
        "Off-target probability shift vs the J background\n"
        "Σ|ΔP| over 20 amino acids per site; mutated sites masked and marked in orange"
    )
    # supxlabel/supylabel landed in matplotlib 3.4; fall back to per-axis labels
    # rather than dying on an older install.
    if hasattr(fig, "supxlabel"):
        fig.supxlabel("Sequence position")
        fig.supylabel("Total absolute shift")
    else:
        for axis in flat[: len(genotype_columns)]:
            axis.set_xlabel("Sequence position")
            axis.set_ylabel("Total absolute shift")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


###############################################################################
# Main
###############################################################################
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.ha2_start is not None and args.ha2_start < 0:
        args.ha2_start = None

    start_header, start_seq, end_header, end_seq, mutations = resolve_mutation_set(args)
    h3_map = build_h3_map(start_seq, args.h3_reference, args.signal_peptide_length, args.ha2_start)

    print(describe_mutation_set(mutations, h3_map, start_header, end_header))

    # Restricting background size keeps a large mutation set tractable: the
    # pairwise matrix only ever needs the root and the single mutants. Orderings
    # walk through every background size, so they cannot be built from a
    # truncated hypercube and are skipped rather than silently approximated.
    pairwise_only = args.max_background_size is not None
    subsets = all_subsets(mutations, args.max_background_size)
    if pairwise_only:
        orderings = []
        print(
            f"Scoring {len(subsets)} genotypes (backgrounds of at most "
            f"{args.max_background_size} mutation(s), out of {2 ** len(mutations)} in the full "
            "hypercube).\n"
            "Ordering/path tables and the exact best/worst-ordering DP need every background "
            "size, so they are SKIPPED for this run. Pairwise epistasis, the per-site "
            "probability table and the off-target shifts are complete.\n"
            "Drop --max-background-size to get the full hypercube and the ordering tables."
        )
    else:
        orderings = all_orderings(mutations, args.max_orders, args.seed)
        print(f"Scoring {len(subsets)} genotypes and reconstructing {len(orderings)} orderings.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    genotype_sequences = {
        genotype_id(subset): apply_mutations(start_seq, subset) for subset in subsets
    }

    pd.DataFrame(
        [
            {
                "mutation": m.name,
                "mutation_h3": m.label(h3_map),
                "position_raw": m.pos,
                "position_h3": h3_map.get(m.pos, ""),
                "wt_aa": m.wt,
                "alt_aa": m.alt,
            }
            for m in mutations
        ]
    ).to_csv(output_dir / "mutations.csv", index=False)

    if orderings:
        pd.DataFrame(
            [
                {
                    "path_id": f"path{i + 1:03d}",
                    "order": order_id(ordering),
                    "order_h3": order_id(ordering, h3_map),
                }
                for i, ordering in enumerate(orderings)
            ]
        ).to_csv(output_dir / "orderings.csv", index=False)

    write_fasta(
        output_dir / "genotypes.fasta",
        [(gid, sequence) for gid, sequence in genotype_sequences.items()],
    )

    metadata = {
        "start_header": start_header,
        "end_header": end_header,
        "mutations": [m.name for m in mutations],
        "mutations_h3": [m.label(h3_map) for m in mutations],
        "n_genotypes": len(subsets),
        "n_genotypes_full_hypercube": 2 ** len(mutations),
        "max_background_size": args.max_background_size,
        "pairwise_only": pairwise_only,
        "n_orderings": len(orderings),
        "base_model": args.base_model,
        "checkpoint_dir": str(args.checkpoint_dir) if args.checkpoint_dir else None,
        "scoring": args.scoring,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

    if args.dry_run:
        print(f"Dry run complete. Enumeration written to {output_dir}")
        return 0

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is not None and str(checkpoint_dir).lower() in {"none", ""}:
        checkpoint_dir = None
    if checkpoint_dir is not None and not Path(checkpoint_dir).exists():
        raise FileNotFoundError(
            f"Checkpoint directory {checkpoint_dir} does not exist. "
            "Pass --checkpoint-dir none to score with the stock --base-model weights."
        )

    model_tag = args.model_tag or (Path(checkpoint_dir).name if checkpoint_dir else args.base_model)
    print(f"Loading {args.base_model} (checkpoint: {checkpoint_dir})...")
    model, device, batch_converter, alphabet, last_layer = load_runtime(args.base_model, checkpoint_dir)
    print(f"Model loaded on {device}; final representation layer {last_layer}.")

    schemes = SCORING_SCHEMES if args.scoring == "both" else (args.scoring,)
    probabilities: Dict[Tuple[str, str], Dict[int, np.ndarray]] = {}
    matrices: Dict[str, np.ndarray] = {}

    gids = list(genotype_sequences)
    sequences = [genotype_sequences[gid] for gid in gids]
    focal_positions = [m.pos for m in mutations]

    if "wt-marginal" in schemes or args.save_matrices:
        print("Computing unmasked (wt-marginal) probability matrices...")
        computed = wt_marginal_matrices(sequences, model, device, batch_converter, alphabet,
                                        args.batch_size)
        for gid, matrix in zip(gids, computed):
            matrices[gid] = matrix
            if "wt-marginal" in schemes:
                probabilities[("wt-marginal", gid)] = {
                    position: matrix[:, position - 1] for position in focal_positions
                }

    if "masked-marginal" in schemes:
        print("Computing masked-marginal probabilities at the mutated sites...")
        for index, gid in enumerate(gids, start=1):
            probabilities[("masked-marginal", gid)] = masked_site_probabilities(
                genotype_sequences[gid], focal_positions, model, device, batch_converter,
                alphabet, args.batch_size,
            )
            print(f"  [{index}/{len(gids)}] {gid}")

    print("Assembling tables...")
    site_table = build_site_table(mutations, subsets, probabilities, h3_map)
    site_table.to_csv(output_dir / "site_probabilities_by_background.csv", index=False)

    steps = summary = extremes = None
    if orderings:
        steps, path_sites, summary = build_path_tables(
            orderings, site_table, h3_map, include_all_sites=not args.no_all_sites_table
        )
        steps.to_csv(output_dir / "order_paths_steps.csv", index=False)
        summary.to_csv(output_dir / "order_paths_summary.csv", index=False)
        if args.no_all_sites_table:
            print("  order_paths_all_sites.csv skipped (--no-all-sites-table); it is a join of "
                  "order_paths_steps.csv onto site_probabilities_by_background.csv.")
        else:
            path_sites.to_csv(output_dir / "order_paths_all_sites.csv", index=False)

        extremes = build_exact_extremes(mutations, site_table, h3_map)
        extremes.to_csv(output_dir / "order_paths_extremes.csv", index=False)

    pairwise = build_pairwise_epistasis(mutations, site_table, h3_map)
    pairwise.to_csv(output_dir / "pairwise_epistasis.csv", index=False)

    for scoring in schemes:
        wide = site_table[site_table["scoring"] == scoring].pivot_table(
            index="mutation_h3", columns="background_h3", values="logit_alt_over_wt"
        )
        wide.to_csv(output_dir / f"logit_by_background_{scoring}.csv")

    if matrices:
        offtarget = build_offtarget_shifts(subsets, matrices, h3_map, start_seq)
        offtarget.to_csv(output_dir / "offtarget_total_abs_shift.csv", index=False)
    else:
        offtarget = None

    if args.save_matrices:
        matrix_dir = output_dir / "probability_matrices"
        matrix_dir.mkdir(parents=True, exist_ok=True)
        for gid, matrix in matrices.items():
            pd.DataFrame(
                matrix, index=AMINO_ACIDS, columns=np.arange(1, matrix.shape[1] + 1)
            ).to_csv(matrix_dir / f"{model_tag}_{gid}.csv")

    if not args.no_plots:
        print("Drawing figures...")
        for scoring in schemes:
            plot_delta_logit_heatmap(site_table, scoring,
                                     output_dir / f"delta_logit_heatmap_{scoring}.png")
            plot_pairwise_heatmap(pairwise, scoring,
                                  output_dir / f"pairwise_epistasis_{scoring}.png")
            plot_odds_with_wt(site_table, scoring,
                              output_dir / f"pairwise_odds_with_wt_{scoring}.png",
                              output_dir / f"pairwise_odds_with_wt_{scoring}.csv")
            plot_delta_odds_vs_wt(site_table, scoring,
                                  output_dir / f"pairwise_delta_odds_{scoring}.png",
                                  output_dir / f"pairwise_delta_odds_{scoring}.csv",
                                  scale=args.delta_odds_scale)
            plot_site_context_spread(site_table, scoring,
                                     output_dir / f"site_context_spread_{scoring}.png")
            if summary is not None:
                plot_path_ranking(steps, summary, scoring,
                                  output_dir / f"path_ranking_{scoring}.png")
        if offtarget is not None:
            plot_offtarget_panel(offtarget, mutations, output_dir / "offtarget_shift_panel.png")

    if summary is not None:
        print("\nTop orderings by Σ log10 P(derived residue):")
        for scoring in schemes:
            best = summary[summary["scoring"] == scoring].head(5)
            print(f"  [{scoring}] (best of {len(orderings)} orderings scored)")
            for _, row in best.iterrows():
                print(f"    {int(row['rank_by_sum_log10_p_alt']):>2}. {row['order_h3']}  "
                      f"Σlog10P={row['sum_log10_p_alt']:.3f}  "
                      f"bottleneck={row['bottleneck_mutation_h3']} at step {row['bottleneck_step']}")

        print("\nExact extremes over ALL orderings (hypercube DP):")
        for _, row in extremes[extremes["metric"] == "sum_log10_p_alt"].iterrows():
            print(f"  [{row['scoring']}] {row['extreme']:>5}: {row['order_h3']}  "
                  f"Σlog10P={row['score']:.3f}  "
                  f"(over {int(row['n_orderings_covered']):,} orderings)")

    print(f"\nWrote outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
