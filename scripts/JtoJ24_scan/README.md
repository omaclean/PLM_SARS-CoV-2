# Ordered-mutation (epistasis) scans between H3N2 HA lineages

Two command-line scans over the mutations separating two H3N2 HA lineages:

| Script | Question |
| --- | --- |
| `epistasis_order_scan.py` | How does the PLM's probability at each mutated site shift depending on which of the other mutations have already fixed? |
| `plant_order_scan.py` | How does the PLANT-predicted 3D position move as the mutations fix, and how much does that trajectory depend on the order? |
| `plot_pairwise_odds.py` | Redraw the pairwise heatmap of any completed run on an odds scale — **raw odds with a WT column** and **Δ odds vs WT**. No model needed. |
| `plot_plant_escape.py` | Redraw the PLANT **single-mutation and pairwise immune-escape** figures for any completed PLANT run. No model needed. |
| `plant_population_escape.py` | Escape scored against the **standing immunity at a date** — the recency-weighted background cloud — instead of against the start lineage alone. No model needed. |
| `make_trial_dataset.py` | Build a small, fast run directory to trial the above on: a per-year subsample of the real cloud, or a planted landscape with a known right answer. |

`run_order_scan.sh` runs the two scans, picking the conda interpreter for each.

## The mutation sets

Both drawn from `Sequences/huH3N2_HA_CDS.translated.fas`. Selectors resolve by
**exact lineage field**, so `--end-id J.2.4` does not pick up `J.2.4.1`.

### J → J.2.4 (`--end-id J.2.4`) — 4 mutations

| Raw | Mutation | H3 (HA1) | Site |
| --- | --- | --- | --- |
| 138 | `N138D` | `N122D` | A |
| 151 | `T151K` | `T135K` | A |
| 205 | `K205R` | `K189R` | B |
| 292 | `K292E` | `K276E` | C/E |

⇒ **16 genotypes, 24 orderings** — the full hypercube runs by default.

### J → J.2.4.1 (`--end-id J.2.4.1`) — 11 mutations

| Raw | Mutation | H3 | | Raw | Mutation | H3 |
| --- | --- | --- | --- | --- | --- | --- |
| 18 | `K18N` | `K2N` | | 189 | `Q189R` | `Q173R` |
| 138 | `N138D` | `N122D` | | 205 | `K205R` | `K189R` |
| 151 | `T151K` | `T135K` | | 292 | `K292E` | `K276E` |
| 160 | `S160N` | `S144N` | | 344 | `T344A` | `T328A` |
| 174 | `N174D` | `N158D` | | 394 | `S394N` | `HA2:S49N` |
| 176 | `I176K` | `I160K` | | | | |

⇒ **2048 genotypes, 39,916,800 orderings**. The J.2.4 set is a strict subset, so
the two runs are directly comparable at those four sites. Run this one with
`--max-background-size 1` (see below) unless you specifically want the full
hypercube.

Raw positions are 1-based into the translated CDS, which carries the 16-residue
signal peptide; H3 HA1 numbering is therefore `raw − 16`. The scripts do not rely
on that offset — they align to `Sequences/H3N2_canonical.fa` via
`create_h3_numbering_map`, so insertions get letter-suffixed labels where the
alignment calls for them. **Check `mutations.csv` in the output directory for the
labels actually used.**

## Keeping a large mutation set simple: `--max-background-size`

`--max-background-size 1` scores only the root and the single mutants — `n+1`
genotypes instead of `2**n`. That is exactly the basis the pairwise epistasis
matrix needs, so **the pairwise matrix and the raw-odds plot are complete, not
approximated**. What it cannot give you is anything that walks through
higher-order backgrounds:

| Still produced | Skipped (needs the full hypercube) |
| --- | --- |
| `site_probabilities_by_background.csv` | `order_paths_steps.csv` |
| `pairwise_epistasis.csv` + heatmap | `order_paths_summary.csv` |
| `pairwise_odds_with_wt_*.png` / `.csv` | `order_paths_all_sites.csv` |
| `offtarget_total_abs_shift.csv` + panel | `order_paths_extremes.csv` |
| `delta_logit_heatmap_*.png` | `path_ranking_*.png` |

The scans print exactly what was skipped and why; nothing is silently truncated.
For the PLANT scan the full mutant is embedded as well even in this mode, because
it defines the start → end axis that the per-mutation displacements are measured
against.

## Exact best/worst ordering without enumerating n!

When the full hypercube *is* scanned, `order_paths_summary.csv` ranks whatever
orderings were enumerated or sampled. `order_paths_extremes.csv` goes further:
an ordering's total score is a sum of per-step terms that each depend only on the
background set, so the best and worst orderings are a longest/shortest path on
the subset lattice, solvable by DP in `O(2**n · n)`. At n = 11 that is ~22k
operations covering all 39,916,800 orderings exactly — the difference between
"best of 2000 sampled" and "best, full stop".

## Why 2**n model runs cover all n! orderings

The `n!` orderings are monotone paths through the mutational hypercube, and they
visit only `2**n` distinct genotypes. Both models are deterministic functions of
the genotype — nothing in them depends on the route taken to reach it — so each
scan evaluates the `2**n` genotypes once and reconstructs every ordering from
that cache. The numbers are identical to running each path end-to-end; it is 16
forward passes instead of 24 × 5 = 120, not an approximation.

## Running

```bash
cd /home3/oml4h/PLM_SARS-CoV-2

# Check the mutation set and hypercube size first. No model is loaded.
./scripts/JtoJ24_scan/run_order_scan.sh --dry-run --epistasis-only

# J -> J.2.4: 4 mutations, full hypercube, both scans
./scripts/JtoJ24_scan/run_order_scan.sh \
  --output-root Results/JtoJ.2.4_scan \
  --env /home3/oml4h/miniconda3/envs/plm_entropy \
  -- --end-id J.2.4

# J -> J.2.4.1: 11 mutations, pairwise matrix only.
# --restrict-to-window is defined only by the PLANT scan (it cannot see the HA2
# substitution), so it goes through --plant-args rather than after `--`.
./scripts/JtoJ24_scan/run_order_scan.sh \
  --output-root Results/JtoJ.2.4.1_scan \
  --env /home3/oml4h/miniconda3/envs/plm_entropy \
  --plant-args "--restrict-to-window" \
  -- --end-id J.2.4.1 --max-background-size 1

# Odds heatmaps for runs that finished before these views existed
# (reads the CSVs; no model). New runs emit them automatically.
/home3/oml4h/miniconda3/envs/plm_entropy/bin/python \
  scripts/JtoJ24_scan/plot_pairwise_odds.py \
  Results/JtoJ.2.4_scan/epistasis Results/JtoJ.2.4.1_scan/epistasis

# Same, for the PLANT single/pairwise escape figures
/home3/oml4h/miniconda3/envs/plm_entropy/bin/python \
  scripts/JtoJ24_scan/plot_plant_escape.py Results/JtoJ.2.4_scan/plant
```

Anything after `--` is forwarded verbatim to **both** scans, so it must be a flag
both accept. `--epistasis-args "..."` and `--plant-args "..."` target one scan
each, for flags only one of them defines.

`plm_entropy` is the only environment on this box carrying fair-esm 2.0.0, a
recent `transformers`, `plotly` **and** `seaborn`, so it runs both scans;
`plm_sars` lacks `seaborn`/`sklearn` and works only for the PLANT scan.

`--dry-run` never loads a model, but the two scans still differ in what they need
to import:

- **`epistasis_order_scan.py --dry-run`** runs anywhere. It tries to build
  canonical H3 labels via `Functions_HuggingFace`, and if that import fails
  (torch/esm absent) it warns and falls back to plain `raw − signal peptide`
  numbering. Everything else is pure Python.
- **`plant_order_scan.py --dry-run`** needs the PLANT environment, because
  validating that all four mutations land inside PLANT's 329-residue HA1 window
  requires PLANT's own reference sequence.

### Conda environments

Environments are addressed by **absolute interpreter path** (`PREFIX/bin/python`),
never `conda activate` — activation does not survive a non-interactive shell or an
sbatch step. This matches `slurm_mutation_acesiblity.sbatch.sh` and
`slurm_prescott_diversity.sbatch.sh`.

| Scan | Needs | Repo convention |
| --- | --- | --- |
| `epistasis_order_scan.py` | `torch`, legacy **fair-esm** (`esm < 3`), `transformers` | `plm_entropy` or `plm_sars` (see `install_test_deps.sh`) |
| `plant_order_scan.py` | `torch`, `transformers`, `plotly`, PLANT's `src/plant` | whichever env already runs `Notebooks/OM_influenza/Plant.run.py` |

The two cannot always share one env: fair-esm and EvolutionaryScale `esm>=3` both
install as the module `esm`, so one shadows the other. Pass `--env` if a single
env does have everything.

With no `--env*` flag the driver falls back to `$CONDA_PREFIX`, then to
`plm_entropy` / `plm_sars` under `~/miniconda3` and `~/anaconda3`, then to the
`python3` on `PATH` with a warning. Edit `CANDIDATE_ENVS` at the top of
`run_order_scan.sh` to make the local names the default.

### Running a scan directly

```bash
/path/to/env/bin/python scripts/JtoJ24_scan/epistasis_order_scan.py \
  --start-id J --end-id J.2.4 \
  --checkpoint-dir /home3/oml4h/hugging_face_downloads/model_weights_topublish/ESM2-HA80 \
  --base-model esm2_t36_3B_UR50D \
  --output-dir Results/JtoJ.2.4_scan/epistasis

/path/to/env/bin/python scripts/JtoJ24_scan/plant_order_scan.py \
  --start-id J --end-id J.2.4 \
  --output-dir Results/JtoJ.2.4_scan/plant
```

Everything else is a flag too — `--help` on either script lists them all. The ones
worth knowing:

| Flag | Effect |
| --- | --- |
| `--fasta`, `--start-id`, `--end-id` | Point the scan at a different pair. Selectors accept a full header, an exact lineage field, a unique substring, or `--start-index`/`--end-index`. |
| `--mutations N138D T151K` | Override the derived set; each is validated against the start sequence. |
| `--max-mutations N` | Keep the first N by position — a fast smoke test. |
| `--max-background-size N` | Cap how many mutations a background may carry. `1` = pairwise matrix only; see above. |
| `--max-orders N` / `--seed` | Sample orderings instead of enumerating `n!`. Required above 8 mutations when scanning the full hypercube. |
| `--base-model`, `--checkpoint-dir`, `--model-tag` | Swap the PLM. `--checkpoint-dir none` scores with stock `--base-model` weights. |
| `--scoring` | `wt-marginal`, `masked-marginal`, or `both` (default). |
| `--plant-model-dir`, `--plant-subfolder`, `--plant-base-model` | Swap the PLANT checkpoint. |
| `--batch-size` | Sequences per forward pass. Drop it if a 3B checkpoint runs out of memory. |
| `--save-matrices` | Dump the full 20 × L probability matrix per genotype. Writes one CSV per genotype, so avoid it on a full large hypercube. |
| `--no-all-sites-table` | Skip `order_paths_all_sites.csv`, which has `orderings × steps × sites × scorings` rows. |
| `--dry-run` | Enumerate and validate only. |

## Scoring schemes

Both are computed by default, because they answer slightly different questions
and disagreement between them is itself informative.

- **`wt-marginal`** — the site is left as-is and the model's distribution at that
  token is read off. This is the convention already used by
  `Notebooks/OM_influenza/Epistasis_hugging_face.py`, so these numbers are
  comparable to existing outputs. The model can see the residue it is scoring,
  which inflates the ancestral state.
- **`masked-marginal`** — the site is replaced with `<mask>` before the forward
  pass. The standard, less self-confirming estimator.

The headline statistic is the **log-odds** `log P(derived) − log P(ancestral)`,
which is invariant to the fact that the 20-amino-acid slice of the model's
vocabulary does not sum to 1. The epistatic shift is that log-odds on a given
background minus its value on the J root.

## Outputs

### `Results/JtoJ.2.4_scan/epistasis/`

| File | Contents |
| --- | --- |
| `mutations.csv` | The mutation set, raw and H3 numbering. Check this first. |
| `orderings.csv` | The 24 orderings and their `path_id`s. |
| `genotypes.fasta` | The 16 full-length genotype sequences. |
| `site_probabilities_by_background.csv` | **The core table.** One row per (scoring, background genotype, mutated site): `p_wt`, `p_alt`, `logit_alt_over_wt`, `delta_logit_vs_root`, `is_fixed`. |
| `logit_by_background_<scoring>.csv` | The same as a mutation × background matrix. |
| `order_paths_steps.csv` | Per ordering, per step: the focal mutation's probability on that step's background. |
| `order_paths_all_sites.csv` | Per ordering, per step: **all four** sites' probabilities on that step's background — how the not-yet-fixed sites get pulled around as the others fix. |
| `order_paths_summary.csv` | Per ordering: Σ log10 P, bottleneck step, rank. |
| `pairwise_epistasis.csv` | Double-mutant-cycle ε on the log-odds scale, both directions plus their asymmetry. |
| `pairwise_odds_with_wt_<scoring>.csv` / `.png` | **Raw odds** `P(derived)/P(ancestral)` per focal mutation on each single-mutant background, with `WT` (unmutated start) as the first column. The Δ-log-odds view measures against WT, so there it is structurally a column of zeros; here it carries the baseline every other column should be read against. Log colour scale, blank diagonal where the focal mutation *is* the context. |
| `pairwise_delta_odds_<scoring>.csv` / `.png` | **Δ odds vs WT**: `odds(focal │ context) − odds(focal │ WT)`. Diverging scale centred at zero and scaled *only* to the deltas. Use this one to read context effects: in the raw-odds view the colour is set by which mutation a row is (N122D's odds sit ~10× above K276E's, so its row saturates), which buries the within-row context signal. WT is not drawn as a column of zeros — its baseline odds appear in the row labels, and the CSV keeps them as a `WT_baseline_odds` column. Colour scale via `--delta-odds-scale`, see below. |
| `order_paths_extremes.csv` | Best and worst ordering over **all** `n!`, from the hypercube DP. Full-hypercube runs only. |
| `offtarget_total_abs_shift.csv` | Σ&#124;ΔP&#124; over the 20 amino acids at **every** position, each genotype vs the J root — epistasis on sites that are not themselves mutated. |
| `delta_logit_heatmap_<scoring>.png` | Focal mutation × background, coloured by epistatic shift. |
| `pairwise_epistasis_<scoring>.png` | Pairwise ε heatmap. |
| `path_ranking_<scoring>.png` | The 24 orderings ranked, with per-step probabilities. |
| `site_context_spread_<scoring>.png` | Spread of each mutation's log-odds across backgrounds. |
| `offtarget_shift_panel.png` | Off-target shift along the sequence, one panel per genotype. |
| `run_metadata.json` | Everything needed to reproduce the run. |

### `Results/JtoJ.2.4_scan/plant/`

| File | Contents |
| --- | --- |
| `genotype_embeddings.csv` | X/Y/Z per genotype, plus distance to J, distance to J.2.4, and position along / deviation from the straight J → J.2.4 axis. |
| `single_mutation_escape.csv` | Per mutation: total antigenic displacement from J, its component along J → J.2.4, the off-axis remainder, and the share of the whole J → J.2.4 move it delivers alone. |
| `pairwise_escape.csv` | Per pair: observed escape, the additive expectation from the two singles, and ε (signed, along the axis) plus the magnitude of the full 3D departure from additivity. |
| `order_paths_steps.csv` | Per ordering, per step: displacement vector, step length, cumulative length, distance to endpoint. |
| `order_paths_summary.csv` | Per ordering: total path length, tortuosity (path ÷ straight line), largest step, max axis deviation, and flags for backtracking. |
| `observed_sequence_embeddings.csv` | The real lineages from the input FASTA in the same space, as reference points. |
| `genotypes_plant_window.fasta` | The 16 genotypes projected onto PLANT's 329-residue HA1 window. |
| `plant_order_paths_3d.html` | Interactive 3D: 16 genotype nodes, 24 path polylines, observed lineages, optional PLANT background cloud. |
| `plant_order_paths_summary.png` | Path-length ranking and approach-to-endpoint curves. Every ordering ends on the same genotype, so for a small set the bars are all nearly the same length — that flatness *is* the result, and the escape figures below are what separates the mutations. |
| `plant_escape_singles_pairs.png` | Each single mutant's antigenic displacement from J, split into the part aimed at J.2.4 and the off-axis remainder; below it, each double mutant's escape against the sum of its two singles. |
| `plant_escape_epistasis_matrix.png` | ε = double − (single a + single b) for every pair, on the J → J.2.4 axis. Blue is sub-additive, red super-additive, matching the PLM heatmaps' orientation. |
| `plant_escape_map.png` | PLANT space rotated so x is the J → J.2.4 axis and y is the off-axis direction carrying the most spread. Each pair's additive prediction (×) is joined to its observed position, so the epistasis is drawn to scale. |

### `Results/JtoJ.2.4_scan/plant/population_escape/`

Written by `plant_population_escape.py`, one set of files per `--as-of DATE`.
Everything above measures escape as displacement from **one stationary point**,
the start lineage. That is not what immunity sees. This subfolder scores each
genotype against the immunity actually standing at a date.

| File | Contents |
| --- | --- |
| `genotype_population_escape_<date>.csv` | Per genotype: `population_escape` (share of the recency-weighted standing immunity it escapes, in [0, 1]) and `escape_gain` vs the start lineage. |
| `single_mutation_population_escape_<date>.csv` | Per mutation: the gain it buys, and that gain as a share of the immunity still covering the start lineage. |
| `pairwise_population_escape_<date>.csv` | Per pair: observed gain, both additive baselines, and ε split into kernel curvature and interaction (see below). |
| `immune_landscape_by_year_<date>.csv` | Where the weight actually sits — weight and sequence count per calendar year. Check this before trusting anything else. |
| `run_metadata_<date>.json` | Kernel, scale, half-life, normalisation, effective sample size, frame check. |
| `population_escape_singles_pairs_<date>.png` | Same layout as `plant_escape_singles_pairs.png`, on the population scale. |
| `population_epistasis_matrix_<date>.png` | ε matrix on the population scale. |
| `immune_landscape_<date>.png` | The weighted immunity as a hexbin density, in the same frame as `plant_escape_map.png`, with the genotypes on top. |
| `population_escape_vs_date.png` | With more than one `--as-of`: one line per mutation, showing how its value changes as the landscape moves. |

**The weighting is a product of two independent choices**, `w(s) = recency(s) ×
share(s)`, because *across-period* and *within-period* sampling bias are
different problems and need different flags.

`recency` is `0.5 ** (age_years / --half-life)`, default half-life **1 year** —
deliberately aggressive, so immunity from two seasons ago counts a quarter as
much. Sequence counts track surveillance effort far more than prevalence (~200
sequences from 1968, tens of thousands from 2022), so `--normalise-by year`
(default) makes each calendar year contribute weight set by *when* it was, not by
how hard anyone was sequencing. `--normalise-by none` drops the periods and
weights every deposited sequence by recency alone.

`share` is how a period splits its weight among **its own** sequences, and
year-normalisation does *not* fix this one:

| `--within-period` | A period's weight follows… | Use when |
| --- | --- | --- |
| `abundance` (default) | its **sampled composition** — a clade sequenced 10× more than another of equal true prevalence carries 10× the immunity | you trust within-year sampling to be roughly proportional to prevalence |
| `unique` | its **antigenic diversity** — each distinct position gets an equal share, split among the sequences sitting on it | you do not, and want duplicate deposits of one phenotype to stop counting twice |
| `density` | diversity on a grid of `--density-radius` — near-identical sequences collapse too | as `unique`, but you also want a tight cluster of 500 similar strains to count like one strain |

`backgrounds.csv` is full of **exact** coordinate ties (identical XYZ across many
rows), so `unique` is a large change rather than a rounding one.

**The default is `abundance` deliberately.** A dense cluster is taken to be a
genuine epidemic and its frequency to be real, so it should carry the immunity
its size implies. PCR error — in vitro and in vivo — puts enough diversity wobble
into the deposited sequences that a real epidemic does not collapse to one point
anyway, which limits how much the abundance/diversity distinction can bite in
practice. `unique` and `density` are there for the opposite case (one lab
depositing the same outbreak repeatedly) and as a sensitivity check; the choice
is recorded in each run's metadata.

**Why the distance kernel saturates.** Cross-immunity to a strain at distance `d`
is `exp(-d / --cross-immunity-scale)`, so escape from it is `1 - exp(-d/scale)`.
That function is steepest at `d = 0`, which is the point — with the default scale
of 2.0 antigenic units:

| Move | Escape from that strain | Gain |
| --- | --- | --- |
| 0.5 units, starting 1.0 away | 0.393 → 0.528 | **+0.135** |
| 1.0 units, starting 10.0 away | 0.993 → 0.996 | +0.004 |

Twice the distance, one thirty-fourth the value. `--kernel sigmoid` puts a plateau
of near-full protection at short range (closer to how HI titres behave, but it
flattens exactly the local gradient this analysis exists to measure);
`--kernel linear` is a hard cutoff at `scale`.

**Most of a pair's ε is not epistasis.** `epistasis = observed − (gain_a +
gain_b)` is additivity *in escape*, and escape is a bounded, saturating function
of distance — so two mutations that add perfectly in coordinates still fail to
add in escape. Scoring the genotype additivity actually predicts, `root + Δa +
Δb`, gives a second baseline and an exact split:

```
epistasis  =  kernel_curvature  +  epistasis_vs_additive_genotype
              (the saturating       (the substitutions
               kernel)               doing something together)
```

Only the second term is epistasis in any useful sense. On the J run as of
2024-01-01, `T135K+K276E` reports ε = +0.036 against an additive expectation of
+0.057 — which looks like a 63% interaction, but that pair is essentially
additive in coordinate space (`epistasis_along_axis` = −0.006), so nearly all of
it is curvature. The closer the immunity sits to the start lineage, the more the
curvature term dominates. Both figures show the split: the dumbbell draws a grey
leg for curvature and a coloured leg for the interaction, and the matrix plots
the interaction alone.

**Are the coordinates comparable?** Yes, and the script checks rather than
assumes: it reports the nearest background sequence to the start lineage and
warns above 1.0 units. For the J run the observed `J.2` embedding
(3.0117, 3.5938, −0.3411) matches background `A/Croatia/HZJZ_6964/2024 [J.2]`
(3.0137, 3.5938, −0.3411) to within fp16 rounding.

**Limits.** `backgrounds.csv` ends in early 2024, so a later `--as-of` quietly
means "all of history" — the script prints the gap. And `--cross-immunity-scale`
sets the units of every number here; it is recorded in each run's metadata for
exactly that reason.

```bash
/home3/oml4h/miniconda3/envs/plm_entropy/bin/python \
  scripts/JtoJ24_scan/plant_population_escape.py \
  Results/JtoJ.2.4_scan/plant \
  --as-of 2023-01-01 --as-of 2023-07-01 --as-of 2024-01-01

# Same, with a year's weight following its antigenic diversity rather than its
# sampled composition
/home3/oml4h/miniconda3/envs/plm_entropy/bin/python \
  scripts/JtoJ24_scan/plant_population_escape.py \
  Results/JtoJ.2.4_scan/plant --as-of 2024-01-01 --within-period unique
```

### Trialling it on something small first

`make_trial_dataset.py` builds a self-contained run directory in seconds.

```bash
PY=/home3/oml4h/miniconda3/envs/plm_entropy/bin/python

# 1. Known-answer check: does the score do what it claims?
$PY scripts/JtoJ24_scan/make_trial_dataset.py --mode synthetic \
      --output-dir Results/JtoJ.2.4_scan/plant_trial_synthetic

# 2. Real data, capped at 250 sequences/year (~10k rows instead of 150k)
$PY scripts/JtoJ24_scan/make_trial_dataset.py \
      --output-dir Results/JtoJ.2.4_scan/plant_trial
```

Each prints the exact follow-up command. The **synthetic** dataset is the one
worth running first: three of its four mutations sit at distance exactly 1.0
from the start lineage on orthogonal axes, so `plant_escape_*` scores all three
identically and *cannot* rank them, while the immunity is placed to one side so
`plant_population_escape.py` must — and the builder prints the order to expect
before you run it. If that order does not come out, the score is wrong, and you
find out without needing a judgement about influenza.

The **subsample** cap is near-harmless under the default `--normalise-by year`,
where each year's total weight comes from recency rather than its sequence count;
the sample is uniform within a year, so composition is preserved in expectation.
Under `--normalise-by none` the count *is* the weight, so re-check any conclusion
drawn there against the full cloud.

## Tests

`tests_jtoj24_scan/` covers all three escape scripts — geometry, kernels,
weighting, both CLIs — with no model, no torch and no network. Ground truth is
closed-form: the synthetic geometry has an endpoint axis of length exactly 3, and
the synthetic immune landscape's weights are exact decimals.

```bash
./tests_jtoj24_scan/run_tests.sh        # fast, offline, fully synthetic
./tests_jtoj24_scan/run_tests.sh -s     # + the real-run / real-background tests
```

## Reading `pairwise_delta_odds`: baseline odds above 1

`WT baseline odds > 1` in a row label means the model already prefers the
*derived* residue over the ancestral one on the unmutated start background. In
J → J.2.4.1 that is true of three sites — `I160K` (191), `HA2:S49N` (117),
`S144N` (17.4) — against ~0.01–0.3 for the rest.

Those rows produce deltas in the tens while everything else moves by ~0.001, so a
linear colour scale renders eight of eleven rows blank white. `--delta-odds-scale`
controls this and nothing else — the plotted quantity is Δ odds either way:

| Value | Behaviour |
| --- | --- |
| `auto` (default) | `symlog` when max │Δ│ exceeds 50× the median │Δ│, else `linear`. The scale actually used is printed in the figure title. |
| `linear` | Equal odds-units get equal colour distance. Correct when baselines are comparable — J → J.2.4 stays on this. |
| `symlog` | Linear near zero, logarithmic beyond. J → J.2.4.1 needs this. |

## PLANT and HA2: `--restrict-to-window`

PLANT scores a fixed 329-residue **HA1** window, so an HA2 substitution is
invisible to it by construction. `plant_order_scan.py` tests every mutation
against that window and **errors by default** rather than embedding a sequence
that silently lacks one — otherwise the missing mutation shows up as a genuine
"this mutation does nothing" result.

For J → J.2.4.1 this fires on `S394N` (`HA2:S49N`). Pass `--restrict-to-window` to
drop out-of-window mutations and scan the remaining 10; what was dropped is
printed and recorded in `run_metadata.json` as `dropped_outside_window`.

## Relationship to the existing PLANT scripts

`plant_order_scan.py` imports `Notebooks/OM_influenza/Plant_batch_fastas.py` and
calls its `load_plant_runtime()` and `embed_dataframe()`, redirecting the module's
path globals from the command line. So there is still one implementation of the
PLANT loading/embedding path in the repo, and the coordinates are directly
comparable to `Notebooks/OM_influenza/Plant.run.py`, which shares the same
checkpoint and pipeline.

`Plant.run.py` itself is not invoked: its input CSV, output directory, reference
and highlighted-lineage lists are hard-coded, and it does a large amount of
unrelated background plotting. Running it 24 times would also reload the model
each time.

PLANT scores a fixed 329-residue HA1 window. All four mutations fall inside it
(H3 122/135/189/276), and `plant_order_scan.py` asserts that every requested
mutation survives the projection before embedding — without that check, a
mutation falling outside the window would produce two identical embeddings and a
trajectory that looks like a genuine "this mutation does nothing" result.

## Reading the results

- **`delta_logit_vs_root` ≈ 0 across all backgrounds** ⇒ that site is
  order-independent; the model sees no epistasis for it.
- **Large positive `delta_logit_vs_root`** ⇒ the background mutations make the
  model *more* willing to accept this one — the classic permissive/potentiating
  pattern, and the reason some orderings rank above others.
- **`order_paths_summary.csv` rank spread** ⇒ if Σ log10 P is nearly flat across
  the 24 orderings, order does not matter much for accessibility. A wide spread
  means the mutations have to arrive in particular sequences.
- **`asymmetry` in `pairwise_epistasis.csv`** ⇒ how far the PLM departs from a
  symmetric energy function. Large values mean ε should not be quoted as a single
  pairwise number.
- **`tortuosity` ≈ 1** in the PLANT summary ⇒ the intermediates lie on the direct
  J → J.2.4 line and the order barely changes the antigenic trajectory. Values
  well above 1, or `backtracks_along_axis = True`, mean some orderings detour
  through antigenic space the others never visit.
- **`frac_of_endpoint` in `single_mutation_escape.csv`** ⇒ the share of the whole
  J → J.2.4 antigenic move one mutation delivers on its own. Values near the
  reciprocal of the mutation count mean the escape is spread evenly; one value
  dominating means a single substitution is carrying the antigenic change and
  the rest are along for the ride.
- **`off_axis_fraction` near or above 0.5** ⇒ that mutation moves the virus
  sideways rather than towards the endpoint. It is still antigenic escape — just
  escape in a direction this particular lineage transition does not explain, so
  it will not show up in anything measured only along the J → J.2.4 axis.
- **`epistasis_along_axis` ≈ 0 for every pair** ⇒ escape is additive and the
  singles predict the doubles; only then is a per-mutation escape number safe to
  quote on its own. `epistasis_magnitude` much larger than
  `|epistasis_along_axis|` means the pair *is* non-additive, but off the axis —
  the double lands somewhere the additive prediction does not, without gaining or
  losing endpoint progress. The `plant_escape_map.png` segments show both cases
  to scale.
