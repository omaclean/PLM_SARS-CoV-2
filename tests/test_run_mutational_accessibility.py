from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_mutational_accessibility.py"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location("run_mutational_accessibility", SCRIPT_PATH)
mut_script = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(mut_script)


def write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for name, sequence in records:
            handle.write(f">{name}\n{sequence}\n")


def write_precomputed_plm(path: Path) -> None:
    matrix = pd.DataFrame(
        [
            ["M", "A", "E"],
            [0.10, 0.60, 0.10],
            [0.10, 0.20, 0.10],
            [0.10, 0.15, 0.10],
            [0.10, 0.10, 0.70],
            [0.70, 0.10, 0.10],
        ],
        index=["sequence", "A", "S", "T", "E", "M"],
    )
    matrix.to_csv(path, header=False)


def build_toy_monthly_guide_inputs(tmp_path: Path) -> dict[str, Path]:
    reference_fasta = tmp_path / "reference.nt.fa"
    diversity_fasta = tmp_path / "diversity.fasta"
    guide_path = tmp_path / "guide.csv"
    plm_path = tmp_path / "toy_plm.csv"
    output_dir = tmp_path / "outputs"

    write_fasta(reference_fasta, [("toy_ref", "ATGGCTGAA")])
    write_fasta(
        diversity_fasta,
        [
            ("seq1", "MAE"),
            ("seq2", "MSE"),
            ("seq3", "MTE"),
        ],
    )
    guide_path.write_text(
        "month,fasta,reference\n"
        f"toy_lineage,{diversity_fasta},{reference_fasta}\n",
        encoding="utf-8",
    )
    write_precomputed_plm(plm_path)

    return {
        "reference_fasta": reference_fasta,
        "diversity_fasta": diversity_fasta,
        "guide_path": guide_path,
        "plm_path": plm_path,
        "output_dir": output_dir,
    }


class TestValidateArgs:
    def test_requires_single_fasta_inputs(self, tmp_path: Path):
        parser = mut_script.build_parser()
        args = parser.parse_args(
            [
                "--analysis-mode",
                "SINGLE_FASTA",
                "--mutation-model",
                "H3N2",
                "--output-dir",
                str(tmp_path / "out"),
                "--precomputed-plm-path",
                str(tmp_path / "plm.csv"),
            ]
        )

        with pytest.raises(ValueError, match="--diversity-fasta is required"):
            mut_script.validate_args(args)

    def test_rejects_conflicting_checkpoint_options(self, tmp_path: Path):
        parser = mut_script.build_parser()
        args = parser.parse_args(
            [
                "--analysis-mode",
                "MONTHLY_GUIDE",
                "--guide-path",
                str(tmp_path / "guide.csv"),
                "--mutation-model",
                "H3N2",
                "--output-dir",
                str(tmp_path / "out"),
                "--model-tag",
                "toy",
                "--base-model",
                "esm-c600m",
                "--model-layer",
                "36",
                "--checkpoint-dir",
                str(tmp_path / "checkpoints"),
                "--checkpoint-glob",
                str(tmp_path / "checkpoints/checkpoint-*"),
            ]
        )

        with pytest.raises(ValueError, match="Provide either --checkpoint-dir or --checkpoint-glob"):
            mut_script.validate_args(args)

    def test_rejects_non_positive_alpha_step(self, tmp_path: Path):
        parser = mut_script.build_parser()
        args = parser.parse_args(
            [
                "--analysis-mode",
                "MONTHLY_GUIDE",
                "--guide-path",
                str(tmp_path / "guide.csv"),
                "--mutation-model",
                "H3N2",
                "--output-dir",
                str(tmp_path / "out"),
                "--precomputed-plm-path",
                str(tmp_path / "plm.csv"),
                "--alpha-step",
                "0",
            ]
        )

        with pytest.raises(ValueError, match="--alpha-step must be > 0"):
            mut_script.validate_args(args)


class TestHelpers:
    def test_parse_scatter_alphas_handles_empty_and_values(self):
        assert mut_script.parse_scatter_alphas("") == []
        assert mut_script.parse_scatter_alphas("-1, 0, 1.5") == [-1.0, 0.0, 1.5]

    def test_infer_epoch_value_extracts_trailing_numeric_token(self):
        assert mut_script.infer_epoch_value("checkpoint-525", 0) == 525.0
        assert mut_script.infer_epoch_value("epoch_12.5", 0) == 12.5
        assert mut_script.infer_epoch_value("final_checkpoint", 7) == 7.0

    def test_format_epoch_tick_label_prefers_epoch_numbers(self):
        assert mut_script._format_epoch_tick_label("raw_model", 0.0) == "raw"
        assert mut_script._format_epoch_tick_label("checkpoint-525", 15.0) == "15"
        assert mut_script._format_epoch_tick_label("epoch_12.5", 7.0) == "7"
        assert mut_script._format_epoch_tick_label("final_checkpoint", 16.0) == "final"

    def test_load_diversity_records_translates_nucleotide_when_requested(self, tmp_path: Path):
        fasta_path = tmp_path / "diversity_nt.fasta"
        write_fasta(fasta_path, [("nt1", "ATGGCTGAA"), ("nt2", "ATGTCTGAA")])

        records, any_nucleotide = mut_script.load_diversity_records(
            fasta_path,
            expect_protein_diversity=False,
            test_mode=False,
            test_max_records=5,
        )

        assert any_nucleotide is True
        assert [str(record.seq) for record in records] == ["MAE", "MSE"]


class TestBuildModelSpecs:
    def test_uses_precomputed_plm_path_directly(self, tmp_path: Path):
        plm_path = tmp_path / "toy_plm.csv"
        plm_path.write_text("sequence,M\nA,1.0\n", encoding="utf-8")
        parser = mut_script.build_parser()
        args = parser.parse_args(
            [
                "--analysis-mode",
                "MONTHLY_GUIDE",
                "--guide-path",
                str(tmp_path / "guide.csv"),
                "--mutation-model",
                "H3N2",
                "--output-dir",
                str(tmp_path / "out"),
                "--precomputed-plm-path",
                str(plm_path),
            ]
        )

        specs = mut_script.build_model_specs(args)

        assert len(specs) == 1
        assert specs[0]["precomputed_plm_path"] == plm_path
        assert specs[0]["epoch_label"] == "toy_plm"

    def test_discovers_child_checkpoint_directories(self, tmp_path: Path):
        checkpoint_root = tmp_path / "model"
        (checkpoint_root / "checkpoint-10").mkdir(parents=True)
        (checkpoint_root / "checkpoint-10" / "model.safetensors").write_text("ckpt-10", encoding="utf-8")
        (checkpoint_root / "checkpoint-2").mkdir(parents=True)
        (checkpoint_root / "checkpoint-2" / "model.safetensors").write_text("ckpt-2", encoding="utf-8")
        (checkpoint_root / "final_checkpoint").mkdir(parents=True)
        (checkpoint_root / "final_checkpoint" / "model.safetensors").write_text("ckpt-10", encoding="utf-8")

        parser = mut_script.build_parser()
        args = parser.parse_args(
            [
                "--analysis-mode",
                "MONTHLY_GUIDE",
                "--guide-path",
                str(tmp_path / "guide.csv"),
                "--mutation-model",
                "H3N2",
                "--output-dir",
                str(tmp_path / "out"),
                "--model-tag",
                "toy",
                "--base-model",
                "esm-c600m",
                "--model-layer",
                "36",
                "--checkpoint-dir",
                str(checkpoint_root),
            ]
        )

        specs = mut_script.build_model_specs(args)

        assert [spec["epoch_label"] for spec in specs] == ["raw_model", "checkpoint-2", "checkpoint-10"]
        assert [spec["epoch_value"] for spec in specs] == [0.0, 1.0, 2.0]

    def test_supports_checkpoint_glob(self, tmp_path: Path):
        checkpoint_root = tmp_path / "model"
        (checkpoint_root / "checkpoint-4").mkdir(parents=True)
        (checkpoint_root / "checkpoint-4" / "model.safetensors").write_text("ckpt-4", encoding="utf-8")
        (checkpoint_root / "checkpoint-8").mkdir(parents=True)
        (checkpoint_root / "checkpoint-8" / "model.safetensors").write_text("ckpt-8", encoding="utf-8")

        parser = mut_script.build_parser()
        args = parser.parse_args(
            [
                "--analysis-mode",
                "MONTHLY_GUIDE",
                "--guide-path",
                str(tmp_path / "guide.csv"),
                "--mutation-model",
                "H3N2",
                "--output-dir",
                str(tmp_path / "out"),
                "--model-tag",
                "toy",
                "--base-model",
                "esm-c600m",
                "--model-layer",
                "36",
                "--checkpoint-glob",
                str(checkpoint_root / "checkpoint-*"),
            ]
        )

        specs = mut_script.build_model_specs(args)

        assert [spec["epoch_label"] for spec in specs] == ["raw_model", "checkpoint-4", "checkpoint-8"]
        assert [spec["epoch_value"] for spec in specs] == [0.0, 1.0, 2.0]


class TestMainErrors:
    def test_main_reports_missing_guide_path(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_mutational_accessibility.py",
                "--analysis-mode",
                "MONTHLY_GUIDE",
                "--mutation-model",
                "H3N2",
                "--output-dir",
                str(tmp_path / "out"),
                "--precomputed-plm-path",
                str(tmp_path / "plm.csv"),
            ],
        )

        with pytest.raises(SystemExit) as exc_info:
            mut_script.main()

        assert exc_info.value.code == 2
        assert "--guide-path is required for MONTHLY_GUIDE mode" in capsys.readouterr().err

    def test_main_reports_missing_guide_file(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path):
        plm_path = tmp_path / "plm.csv"
        write_precomputed_plm(plm_path)
        missing_guide = tmp_path / "missing_guide.csv"

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_mutational_accessibility.py",
                "--analysis-mode",
                "MONTHLY_GUIDE",
                "--guide-path",
                str(missing_guide),
                "--mutation-model",
                "H3N2",
                "--output-dir",
                str(tmp_path / "out"),
                "--precomputed-plm-path",
                str(plm_path),
            ],
        )

        with pytest.raises(SystemExit) as exc_info:
            mut_script.main()

        assert exc_info.value.code == 2
        assert f"Guide file not found: {missing_guide}" in capsys.readouterr().err


class TestIntegration:
    def test_run_analysis_monthly_guide_with_precomputed_plm(self, tmp_path: Path):
        inputs = build_toy_monthly_guide_inputs(tmp_path)
        parser = mut_script.build_parser()
        args = parser.parse_args(
            [
                "--analysis-mode",
                "MONTHLY_GUIDE",
                "--guide-path",
                str(inputs["guide_path"]),
                "--mutation-model",
                "H3N2",
                "--output-dir",
                str(inputs["output_dir"]),
                "--expect-protein-diversity",
                "--precomputed-plm-path",
                str(inputs["plm_path"]),
                "--alpha-start",
                "0",
                "--alpha-stop",
                "0",
                "--alpha-step",
                "1",
                "--scatter-alphas",
                "0",
                "--scatter-max-points",
                "1000",
                "--no-alpha-parallel",
            ]
        )

        mut_script.validate_args(args)
        exit_code = mut_script.run_analysis(args)

        assert exit_code == 0
        combined_path = inputs["output_dir"] / "tables" / "combined_long_table.csv"
        status_path = inputs["output_dir"] / "tables" / "model_run_status.tsv"
        manifest_path = inputs["output_dir"] / "run_manifest.json"
        plots_dir = inputs["output_dir"] / "plots"
        plot_path = plots_dir / "epoch_metric_summary.png"
        focused_alpha_plot_path = plots_dir / "alpha_sweep_metrics_selected.png"
        epoch_logistic_plot_path = plots_dir / "epoch_metric_logistic.png"
        epoch_spearman_plot_path = plots_dir / "epoch_metric_spearman_plm_vs_mut.png"
        per_model_plot_path = plots_dir / "per_model" / "toy_plm" / "alpha_sweep_metrics_selected.png"

        assert combined_path.exists()
        assert manifest_path.exists()
        assert status_path.exists()
        assert plot_path.exists()
        assert focused_alpha_plot_path.exists()
        assert epoch_logistic_plot_path.exists()
        assert epoch_spearman_plot_path.exists()
        assert per_model_plot_path.exists()

        combined_df = pd.read_csv(combined_path)
        status_df = pd.read_csv(status_path, sep="\t")

        assert not combined_df.empty
        assert set(["model", "epoch_label", "lineage", "plm_prob", "mut_prob", "obs_freq"]).issubset(combined_df.columns)
        assert (status_df["status"] == "completed").any()
import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_mutational_accessibility.py"


def _load_script_module():
    module_name = "run_mutational_accessibility"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


rma = _load_script_module()


def _make_args(**overrides):
    base = dict(
        analysis_mode="MONTHLY_GUIDE",
        reference_fasta=None,
        diversity_fasta=None,
        guide_path=Path("guide.csv"),
        label="population",
        mutation_model="H3N2",
        output_dir=Path("out"),
        expect_protein_diversity=False,
        plm_max_aa_length=None,
        plm_max_nt_length=None,
        filter_fixed_mutations=True,
        filter_singleton_mutations=False,
        skip_low_count_sites=False,
        min_obs_count=2,
        alpha_start=-1.0,
        alpha_stop=1.0,
        alpha_step=0.1,
        alpha_parallel=True,
        alpha_sweep_min_grid=8,
        alpha_sweep_max_workers=None,
        scatter_alphas="-1,0,1",
        scatter_max_points=200000,
        test_mode=False,
        test_max_targets=1,
        test_max_records=5,
        precomputed_plm_path=None,
        model_tag="ESMC_600M_FLU",
        base_model="esm-c600m",
        model_layer=36,
        checkpoint_dir=Path("checkpoint-root"),
        checkpoint_glob=None,
        force_recompute_plm=False,
        gpu_required=True,
        mutation_baseline_x=-2.0,
        regen_figures_only=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _install_fake_functions_hf(monkeypatch, **attrs):
    fake_module = types.ModuleType("Functions_HuggingFace")
    for name, value in attrs.items():
        setattr(fake_module, name, value)
    monkeypatch.setitem(sys.modules, "Functions_HuggingFace", fake_module)
    return fake_module


class TestValidateArgs:
    def test_single_fasta_requires_diversity_and_reference(self):
        args = _make_args(analysis_mode="SINGLE_FASTA", diversity_fasta=None, reference_fasta=None)
        with pytest.raises(ValueError, match="--diversity-fasta is required"):
            rma.validate_args(args)

        args = _make_args(
            analysis_mode="SINGLE_FASTA",
            diversity_fasta=Path("diversity.fasta"),
            reference_fasta=None,
        )
        with pytest.raises(ValueError, match="--reference-fasta is required"):
            rma.validate_args(args)

    def test_checkpoint_backed_runs_require_model_fields(self):
        args = _make_args(model_tag=None, base_model=None, model_layer=None)
        with pytest.raises(ValueError, match="--model-tag, --base-model, --model-layer"):
            rma.validate_args(args)

    def test_rejects_checkpoint_dir_and_glob_together(self):
        args = _make_args(checkpoint_dir=Path("a"), checkpoint_glob="checkpoint-*")
        with pytest.raises(ValueError, match="either --checkpoint-dir or --checkpoint-glob"):
            rma.validate_args(args)

    def test_rejects_non_positive_alpha_step(self):
        args = _make_args(alpha_step=0.0)
        with pytest.raises(ValueError, match="--alpha-step must be > 0"):
            rma.validate_args(args)

    def test_regen_figures_only_skips_model_validation(self):
        args = _make_args(
            analysis_mode=None,
            mutation_model=None,
            model_tag=None,
            base_model=None,
            model_layer=None,
            checkpoint_dir=None,
            regen_figures_only=True,
        )

        rma.validate_args(args)


class TestParsingHelpers:
    def test_parse_alpha_grid_is_inclusive(self):
        args = _make_args(alpha_start=-0.2, alpha_stop=0.2, alpha_step=0.2)
        grid = rma.parse_alpha_grid(args)
        np.testing.assert_allclose(grid, np.array([-0.2, 0.0, 0.2]))

    def test_parse_scatter_alphas_handles_empty_and_values(self):
        assert rma.parse_scatter_alphas("") == []
        assert rma.parse_scatter_alphas("-1, 0, 1.5") == [-1.0, 0.0, 1.5]

    def test_normalise_plm_matrix_drops_sequence_row(self):
        raw_df = pd.DataFrame(
            [["A", "C"], ["0.1", "0.2"], ["0.3", "0.4"]],
            index=["sequence", "A", "C"],
            columns=[1, 2],
        )
        normalised = rma.normalise_plm_matrix(raw_df)
        assert list(normalised.index) == ["A", "C"]
        assert float(normalised.loc["A", 1]) == pytest.approx(0.1)

    def test_normalise_plm_matrix_drops_non_canonical_token_rows(self):
        raw_df = pd.DataFrame(
            [["A", "C"], ["0.1", "0.2"], ["0.3", "0.4"], ["0.5", "0.6"], ["0.7", "0.8"]],
            index=["sequence", "A", "C", "<mask>", "X"],
            columns=[1, 2],
        )

        normalised = rma.normalise_plm_matrix(raw_df)

        assert list(normalised.index) == ["A", "C"]

    def test_infer_epoch_value_uses_last_numeric_token(self):
        assert rma.infer_epoch_value("checkpoint-525", 0) == 525.0
        assert rma.infer_epoch_value("epoch_3.5_snapshot", 0) == 3.5
        assert rma.infer_epoch_value("final_checkpoint", 7) == 7.0

    def test_fit_logistic_site_correlation_returns_finite_fit_for_near_separated_signal(self):
        score_values = pd.Series([1e-10, 2e-10, 5e-10, 1e-8, 2e-8, 5e-8, 1e-6, 2e-6, 5e-6, 1e-4])
        binary_outcome = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        corr, intercept, slope = rma.fit_logistic_site_correlation(score_values, binary_outcome)

        assert np.isfinite(corr)
        assert np.isfinite(intercept)
        assert np.isfinite(slope)
        assert corr > 0

    def test_logistic_feature_model_matches_site_helper_when_given_same_site_level_question(self):
        score_values = pd.Series([1e-10, 2e-10, 5e-10, 1e-8, 2e-8, 5e-8, 1e-6, 2e-6, 5e-6, 1e-4])
        binary_outcome = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        site_corr, _, _ = rma.fit_logistic_site_correlation(score_values, binary_outcome)
        feature_result = rma.fit_logistic_feature_model(
            pd.DataFrame({"log10_score": np.log10(score_values)}),
            binary_outcome,
        )

        assert feature_result["logistic_fitted_prob_corr"] == pytest.approx(site_corr, abs=1e-6)

    def test_logistic_feature_model_can_enforce_non_negative_coefficients(self):
        feature_frame = pd.DataFrame({"score": np.array([1, 2, 3, 4, 5, 6], dtype=float)})
        binary_outcome = pd.Series([1, 1, 1, 0, 0, 0], dtype=float)

        unconstrained = rma.fit_logistic_feature_model(feature_frame, binary_outcome)
        constrained = rma.fit_logistic_feature_model(
            feature_frame,
            binary_outcome,
            enforce_positive_coefficients=True,
        )

        assert float(unconstrained["logistic_coef_score"]) < 0.0
        assert float(constrained["logistic_coef_score"]) >= 0.0
        assert constrained["logistic_constraint"] == "non_negative_coefficients"

    def test_site_logistic_and_hurdle_logistic_diverge_when_row_definitions_differ(self):
        combined_df = pd.DataFrame(
            [
                {"lineage": "lin1", "position": 1, "ref_aa": "A", "aa": "C", "plm_prob": 0.90, "mut_prob": 0.70, "obs_freq": 0.20, "obs_present": 1},
                {"lineage": "lin1", "position": 1, "ref_aa": "A", "aa": "G", "plm_prob": 0.80, "mut_prob": 0.20, "obs_freq": 0.00, "obs_present": 0},
                {"lineage": "lin1", "position": 2, "ref_aa": "T", "aa": "C", "plm_prob": 0.70, "mut_prob": 0.60, "obs_freq": 0.15, "obs_present": 1},
                {"lineage": "lin1", "position": 2, "ref_aa": "T", "aa": "G", "plm_prob": 0.60, "mut_prob": 0.20, "obs_freq": 0.00, "obs_present": 0},
                {"lineage": "lin1", "position": 3, "ref_aa": "L", "aa": "C", "plm_prob": 0.65, "mut_prob": 0.25, "obs_freq": 0.00, "obs_present": 0},
                {"lineage": "lin1", "position": 3, "ref_aa": "L", "aa": "G", "plm_prob": 0.50, "mut_prob": 0.15, "obs_freq": 0.00, "obs_present": 0},
                {"lineage": "lin1", "position": 4, "ref_aa": "V", "aa": "C", "plm_prob": 0.40, "mut_prob": 0.20, "obs_freq": 0.00, "obs_present": 0},
                {"lineage": "lin1", "position": 4, "ref_aa": "V", "aa": "G", "plm_prob": 0.30, "mut_prob": 0.10, "obs_freq": 0.00, "obs_present": 0},
            ]
        )

        site_df = (
            combined_df.groupby(["lineage", "position", "ref_aa"], as_index=False)
            .agg(site_score=("plm_prob", "max"), site_mutated=("obs_present", "max"))
        )
        site_corr, _, _ = rma.fit_logistic_site_correlation(site_df["site_score"], site_df["site_mutated"])

        base_df = rma.build_hurdle_base_frame(combined_df)
        hurdle_logistic = rma.fit_logistic_feature_model(
            pd.DataFrame({"combined_score": base_df["log_plm_prob"]}),
            base_df["obs_present"],
        )

        assert np.isfinite(site_corr)
        assert np.isfinite(hurdle_logistic["logistic_fitted_prob_corr"])
        assert hurdle_logistic["logistic_fitted_prob_corr"] != pytest.approx(site_corr, abs=1e-3)


class TestCheckpointDiscovery:
    def test_discover_checkpoint_dirs_finds_only_safetensor_children(self, tmp_path):
        for name, text in [("checkpoint-10", "ckpt-10"), ("checkpoint-2", "ckpt-2"), ("final_checkpoint", "ckpt-10")]:
            path = tmp_path / name
            path.mkdir()
            (path / "model.safetensors").write_text(text)
        (tmp_path / "notes").mkdir()

        discovered = rma._discover_checkpoint_dirs(tmp_path)
        assert [path.name for path in discovered] == ["checkpoint-2", "checkpoint-10", "final_checkpoint"]

    def test_build_model_specs_from_parent_checkpoint_directory(self, tmp_path):
        for name, text in [("checkpoint-35", "ckpt-35"), ("checkpoint-105", "ckpt-105"), ("final_checkpoint", "ckpt-105")]:
            path = tmp_path / name
            path.mkdir()
            (path / "model.safetensors").write_text(text)

        args = _make_args(checkpoint_dir=tmp_path, checkpoint_glob=None, model_tag="ESMC_600M_FLU")
        specs = rma.build_model_specs(args)

        assert [spec["epoch_label"] for spec in specs] == ["raw_model", "checkpoint-35", "checkpoint-105"]
        assert specs[0]["model_tag"] == "ESMC_600M_FLU_raw"
        assert specs[2]["epoch_value"] == 2.0

    def test_build_model_specs_from_glob(self, tmp_path):
        for name, text in [("checkpoint-4", "ckpt-4"), ("checkpoint-12", "ckpt-12")]:
            path = tmp_path / name
            path.mkdir()
            (path / "model.safetensors").write_text(text)

        args = _make_args(
            checkpoint_dir=None,
            checkpoint_glob=str(tmp_path / "checkpoint-*"),
            model_tag="ESMC_600M_FLU",
        )
        specs = rma.build_model_specs(args)
        assert [spec["epoch_label"] for spec in specs] == ["raw_model", "checkpoint-4", "checkpoint-12"]
        assert [spec["epoch_value"] for spec in specs] == [0.0, 1.0, 2.0]

    def test_build_model_specs_single_checkpoint_fallback(self, tmp_path):
        checkpoint_dir = tmp_path / "final_checkpoint"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "model.safetensors").write_text("final-only")

        args = _make_args(checkpoint_dir=checkpoint_dir, checkpoint_glob=None, model_tag="ESMC_600M_FLU")
        specs = rma.build_model_specs(args)

        assert len(specs) == 1
        assert specs[0]["epoch_label"] == "final_checkpoint"
        assert specs[0]["checkpoint_dir"] == checkpoint_dir


class TestLoadDiversityRecords:
    def test_translates_nucleotide_records_when_not_expecting_protein(self, tmp_path, monkeypatch):
        fasta_path = tmp_path / "diversity.fasta"
        fasta_path.write_text(">seq1\nATGGCC\n>seq2\nATGAAA\n")

        _install_fake_functions_hf(
            monkeypatch,
            _is_probably_nucleotide_sequence=lambda sequence: True,
        )

        processed, any_nucleotide = rma.load_diversity_records(
            fasta_path,
            expect_protein_diversity=False,
            test_mode=False,
            test_max_records=5,
        )

        assert any_nucleotide is True
        assert [str(record.seq) for record in processed] == ["MA", "MK"]

    def test_preserves_records_when_expecting_protein(self, tmp_path, monkeypatch):
        fasta_path = tmp_path / "diversity.fasta"
        fasta_path.write_text(">seq1\nATGGCC\n")

        _install_fake_functions_hf(
            monkeypatch,
            _is_probably_nucleotide_sequence=lambda sequence: True,
        )

        processed, any_nucleotide = rma.load_diversity_records(
            fasta_path,
            expect_protein_diversity=True,
            test_mode=False,
            test_max_records=5,
        )

        assert any_nucleotide is True
        assert str(processed[0].seq) == "ATGGCC"


class TestRowBuilding:
    def test_build_combined_rows_filters_fixed_mutations(self):
        plm_matrix = pd.DataFrame([[0.0], [0.25]], index=["A", "C"], columns=[1])
        lineage_data = {
            "coord_map": {0: 0},
            "full_ref_protein": "A",
            "mut_profile": pd.DataFrame({1: {"A": 0.0, "C": 0.2}}),
            "obs_freq": pd.DataFrame({1: {"A": 0.0, "C": 1.0}}),
            "obs_depth": {1: 12},
        }
        args = _make_args(filter_fixed_mutations=True)
        model_spec = {"model_tag": "m1", "epoch_label": "checkpoint-1", "epoch_value": 1.0}

        rows = rma.build_combined_rows(args, model_spec, "lin1", lineage_data, plm_matrix)
        assert rows == []

    def test_build_combined_rows_zeroes_singletons_when_requested(self):
        plm_matrix = pd.DataFrame([[0.0], [0.25]], index=["A", "C"], columns=[1])
        lineage_data = {
            "coord_map": {0: 0},
            "full_ref_protein": "A",
            "mut_profile": pd.DataFrame({1: {"A": 0.0, "C": 0.2}}),
            "obs_freq": pd.DataFrame({1: {"A": 0.0, "C": 0.2}}),
            "obs_depth": {1: 3},
        }
        args = _make_args(filter_singleton_mutations=True, min_obs_count=2)
        model_spec = {"model_tag": "m1", "epoch_label": "checkpoint-1", "epoch_value": 1.0}

        rows = rma.build_combined_rows(args, model_spec, "lin1", lineage_data, plm_matrix)
        assert len(rows) == 1
        assert rows[0]["obs_freq"] == 0.0
        assert rows[0]["obs_present"] == 0

    def test_build_combined_rows_skips_low_count_sites_when_requested(self):
        plm_matrix = pd.DataFrame([[0.0], [0.25]], index=["A", "C"], columns=[1])
        lineage_data = {
            "coord_map": {0: 0},
            "full_ref_protein": "A",
            "mut_profile": pd.DataFrame({1: {"A": 0.0, "C": 0.2}}),
            "obs_freq": pd.DataFrame({1: {"A": 0.0, "C": 0.2}}),
            "obs_depth": {1: 3},
        }
        args = _make_args(filter_singleton_mutations=True, skip_low_count_sites=True, min_obs_count=2)
        model_spec = {"model_tag": "m1", "epoch_label": "checkpoint-1", "epoch_value": 1.0}

        rows = rma.build_combined_rows(args, model_spec, "lin1", lineage_data, plm_matrix)
        assert rows == []

    def test_warn_on_excess_mutation_rows_does_not_warn_at_or_below_20x_sites(self):
        combined_df = pd.DataFrame(
            [
                {"lineage": "lin1", "position": 1, "ref_aa": "A", "aa": f"X{i}"}
                for i in range(20)
            ]
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rma.warn_on_excess_mutation_rows(combined_df, context_label="test")

        assert caught == []

    def test_warn_on_excess_mutation_rows_warns_above_20x_sites(self):
        combined_df = pd.DataFrame(
            [
                {"lineage": "lin1", "position": 1, "ref_aa": "A", "aa": f"X{i}"}
                for i in range(21)
            ]
        )

        with pytest.warns(UserWarning, match=r"exceeding the hard limit of 20x sites"):
            rma.warn_on_excess_mutation_rows(combined_df, context_label="test")


class TestAlignmentAndCorrelations:
    def test_resolve_plm_coordinate_maps_handles_free_end_gap_alignment(self):
        args = _make_args(use_global_plm_reference=True)
        lineage_data = {
            "coord_map": {0: 5, 1: 6, 2: 8, 3: 9},
            "plm_ref_protein": "ACDE",
        }

        resolved_map, global_to_lineage_trim, alignment = rma.resolve_plm_coordinate_maps(
            args,
            "AQCDE",
            lineage_data,
        )

        assert resolved_map == {1: 5, 2: 6, 3: 8, 4: 9}
        assert global_to_lineage_trim == {1: 0, 2: 1, 3: 2, 4: 3}
        assert alignment is not None

    def test_remapped_alignment_preserves_perfect_correlations(self):
        args = _make_args(use_global_plm_reference=True)
        model_spec = {"model_tag": "m1", "epoch_label": "checkpoint-1", "epoch_value": 1.0}
        lineage_data = {
            "coord_map": {0: 0, 1: 1, 2: 2, 3: 3},
            "full_ref_protein": "ACDE",
            "plm_ref_protein": "ACDE",
            "mut_profile": pd.DataFrame({
                1: {"Y": 0.8},
                2: {"Y": 0.5},
                3: {"Y": 0.2},
                4: {"Y": 0.0},
            }),
            "obs_freq": pd.DataFrame({
                1: {"Y": 0.8},
                2: {"Y": 0.5},
                3: {"Y": 0.2},
                4: {"Y": 0.0},
            }),
            "obs_depth": {1: 10, 2: 10, 3: 10, 4: 10},
        }
        plm_matrix = pd.DataFrame([[0.13, 0.8, 0.5, 0.2, 0.0]], index=["Y"], columns=[1, 2, 3, 4, 5])

        resolved_map, _, _ = rma.resolve_plm_coordinate_maps(args, "AQCDE", lineage_data)
        rows = rma.build_combined_rows(
            args,
            model_spec,
            "lin1",
            lineage_data,
            plm_matrix,
            coord_map=resolved_map,
        )
        combined_df = pd.DataFrame(rows)
        lineage_metrics = rma.compute_epoch_lineage_metrics(combined_df)
        summary = rma.summarize_epoch_metrics(lineage_metrics)

        assert combined_df["position"].tolist() == [1, 2, 3, 4]
        assert combined_df["plm_prob"].tolist() == pytest.approx([0.8, 0.5, 0.2, 0.0])
        assert combined_df["mut_prob"].tolist() == pytest.approx([0.8, 0.5, 0.2, 0.0])
        assert combined_df["obs_freq"].tolist() == pytest.approx([0.8, 0.5, 0.2, 0.0])
        assert lineage_metrics.loc[0, "spearman_obs_freq_vs_plm"] == pytest.approx(1.0)
        assert lineage_metrics.loc[0, "pearson_obs_freq_vs_plm"] == pytest.approx(1.0)
        assert lineage_metrics.loc[0, "spearman_obs_freq_vs_mut_baseline"] == pytest.approx(1.0)
        assert lineage_metrics.loc[0, "pearson_obs_freq_vs_mut_baseline"] == pytest.approx(1.0)
        assert lineage_metrics.loc[0, "spearman_plm_vs_mut"] == pytest.approx(1.0)
        assert lineage_metrics.loc[0, "pearson_plm_vs_mut"] == pytest.approx(1.0)
        assert summary.loc[0, "spearman_obs_freq_vs_plm"] == pytest.approx(1.0)
        assert summary.loc[0, "pearson_obs_freq_vs_plm"] == pytest.approx(1.0)
        assert summary.loc[0, "pearson_plm_vs_mut"] == pytest.approx(1.0)


class TestMetricSummaries:
    def test_compute_epoch_lineage_metrics_and_summary(self):
        combined_df = pd.DataFrame(
            [
                {"model": "m1", "epoch_label": "checkpoint-1", "epoch_value": 1.0, "lineage": "lin1", "position": 1, "ref_aa": "A", "aa": "C", "plm_prob": 0.8, "mut_prob": 0.4, "obs_freq": 0.4, "obs_present": 1, "depth": 10.0},
                {"model": "m1", "epoch_label": "checkpoint-1", "epoch_value": 1.0, "lineage": "lin1", "position": 2, "ref_aa": "A", "aa": "G", "plm_prob": 0.2, "mut_prob": 0.1, "obs_freq": 0.0, "obs_present": 0, "depth": 10.0},
                {"model": "m1", "epoch_label": "checkpoint-2", "epoch_value": 2.0, "lineage": "lin1", "position": 1, "ref_aa": "A", "aa": "C", "plm_prob": 0.7, "mut_prob": 0.3, "obs_freq": 0.3, "obs_present": 1, "depth": 10.0},
                {"model": "m1", "epoch_label": "checkpoint-2", "epoch_value": 2.0, "lineage": "lin1", "position": 2, "ref_aa": "A", "aa": "G", "plm_prob": 0.1, "mut_prob": 0.05, "obs_freq": 0.0, "obs_present": 0, "depth": 10.0},
            ]
        )

        lineage_metrics = rma.compute_epoch_lineage_metrics(combined_df)
        summary = rma.summarize_epoch_metrics(lineage_metrics)

        assert len(lineage_metrics) == 2
        assert list(summary["epoch_label"]) == ["checkpoint-1", "checkpoint-2"]

    def test_alpha_sweep_exports_unique_and_pooled_site_counts(self, monkeypatch):
        import Functions_HuggingFace as fhf

        combined_df = pd.DataFrame(
            [
                {"lineage": "lin1", "position": 1, "ref_aa": "A", "aa": "C", "plm_prob": 0.8, "mut_prob": 0.8, "obs_freq": 0.4, "obs_present": 1},
                {"lineage": "lin1", "position": 1, "ref_aa": "A", "aa": "G", "plm_prob": 0.2, "mut_prob": 0.2, "obs_freq": 0.0, "obs_present": 0},
                {"lineage": "lin2", "position": 1, "ref_aa": "A", "aa": "C", "plm_prob": 0.7, "mut_prob": 0.7, "obs_freq": 0.3, "obs_present": 1},
                {"lineage": "lin2", "position": 1, "ref_aa": "A", "aa": "G", "plm_prob": 0.3, "mut_prob": 0.3, "obs_freq": 0.0, "obs_present": 0},
                {"lineage": "lin2", "position": 2, "ref_aa": "T", "aa": "C", "plm_prob": 0.6, "mut_prob": 0.6, "obs_freq": 0.2, "obs_present": 1},
                {"lineage": "lin2", "position": 2, "ref_aa": "T", "aa": "G", "plm_prob": 0.4, "mut_prob": 0.4, "obs_freq": 0.0, "obs_present": 0},
                {"lineage": "lin1", "position": 3, "ref_aa": "V", "aa": "C", "plm_prob": 0.9, "mut_prob": 0.9, "obs_freq": 0.0, "obs_present": 0},
                {"lineage": "lin1", "position": 3, "ref_aa": "V", "aa": "G", "plm_prob": 0.1, "mut_prob": 0.1, "obs_freq": 0.0, "obs_present": 0},
            ]
        )

        result = fhf.evaluate_alpha_sweep(combined_df, np.array([0.0]), parallel=False, pseudocount=1e-16)

        assert result.loc[0, "n_sites_used"] == 2
        assert result.loc[0, "n_pooled_lineage_sites_used"] == 3

    def test_evaluate_alpha_sweep_by_lineage_averages_metrics_across_lineages(self, monkeypatch):
        def fake_evaluate_alpha_sweep(df, alpha_grid, **kwargs):
            lineage_name = str(df["lineage"].iloc[0])
            metric_value = 0.2 if lineage_name == "lin1" else 0.8
            return pd.DataFrame(
                {
                    "alpha": [0.0],
                    "site_top10pct_mutated_enrichment": [metric_value],
                    "site_top10pct_mutated_precision": [metric_value],
                    "site_rank_spearman_r": [metric_value],
                    "mut_flat_global_spearman_r": [metric_value],
                    "mut_flat_global_pearson_r": [metric_value],
                    "mut_flat_mean_site_nll": [1.0 - metric_value],
                }
            )

        _install_fake_functions_hf(monkeypatch, evaluate_alpha_sweep=fake_evaluate_alpha_sweep)

        combined_df = pd.DataFrame(
            [
                {"lineage": "lin1", "position": 1, "ref_aa": "A", "aa": "C", "plm_prob": 0.8, "mut_prob": 0.7, "obs_freq": 0.4, "obs_present": 1, "depth": 10.0},
                {"lineage": "lin1", "position": 1, "ref_aa": "A", "aa": "G", "plm_prob": 0.2, "mut_prob": 0.3, "obs_freq": 0.0, "obs_present": 0, "depth": 10.0},
                {"lineage": "lin2", "position": 1, "ref_aa": "A", "aa": "C", "plm_prob": 0.7, "mut_prob": 0.6, "obs_freq": 0.2, "obs_present": 1, "depth": 10.0},
                {"lineage": "lin2", "position": 1, "ref_aa": "A", "aa": "G", "plm_prob": 0.3, "mut_prob": 0.4, "obs_freq": 0.0, "obs_present": 0, "depth": 10.0},
            ]
        )

        lineage_alpha_df = rma.evaluate_alpha_sweep_by_lineage(combined_df, np.array([0.0]), parallel=False, pseudocount=1e-16)
        summary_df = rma.summarize_lineage_metric_table(
            lineage_alpha_df,
            group_cols=["alpha", "alpha_label", "model_variant", "is_mutation_only_baseline", "input_score_formula"],
        )
        plm_rows = summary_df.loc[summary_df["model_variant"] == "plm_alpha_sweep"]

        assert set(lineage_alpha_df["lineage"]) == {"lin1", "lin2"}
        assert len(plm_rows) == 1
        assert float(plm_rows.iloc[0]["mut_flat_global_spearman_r"]) == pytest.approx(0.5)
        assert int(plm_rows.iloc[0]["n_lineages_averaged"]) == 2

    def test_evaluate_logistic_alpha_sweep_by_lineage_returns_populated_metrics(self):
        combined_df = pd.DataFrame(
            [
                {"lineage": "lin1", "plm_prob": 0.90, "mut_prob": 0.80, "obs_present": 1},
                {"lineage": "lin1", "plm_prob": 0.70, "mut_prob": 0.60, "obs_present": 1},
                {"lineage": "lin1", "plm_prob": 0.20, "mut_prob": 0.30, "obs_present": 0},
                {"lineage": "lin1", "plm_prob": 0.10, "mut_prob": 0.20, "obs_present": 0},
            ]
        )

        logistic_df = rma.evaluate_logistic_alpha_sweep_by_lineage(combined_df, np.array([0.0, 1.0]))

        assert list(logistic_df["alpha"]) == [0.0, 1.0]
        assert logistic_df["site_logistic_auroc"].notna().all()
        assert logistic_df["site_logistic_pr_auc"].notna().all()


class TestRunAnalysisSmoke:
    def test_run_analysis_regen_figures_only_uses_existing_tables(self, tmp_path, monkeypatch):
        output_dir = tmp_path / "out"
        tables_dir = output_dir / "tables"
        plots_dir = output_dir / "plots"
        tables_dir.mkdir(parents=True)
        plots_dir.mkdir(parents=True)

        combined_df = pd.DataFrame(
            [
                {"model": "toy", "epoch_label": "raw_model", "epoch_value": 0.0, "lineage": "lin1", "position": 1, "ref_aa": "A", "aa": "C", "plm_prob": 0.8, "mut_prob": 0.6, "obs_freq": 0.3, "obs_present": 1, "depth": 10.0},
                {"model": "toy", "epoch_label": "raw_model", "epoch_value": 0.0, "lineage": "lin1", "position": 1, "ref_aa": "A", "aa": "G", "plm_prob": 0.2, "mut_prob": 0.2, "obs_freq": 0.0, "obs_present": 0, "depth": 10.0},
            ]
        )
        alpha_df = pd.DataFrame(
            [
                {
                    "model": "toy",
                    "epoch_label": "raw_model",
                    "epoch_value": 0.0,
                    "alpha": 0.0,
                    "alpha_label": "0",
                    "model_variant": "plm_alpha_sweep",
                    "is_mutation_only_baseline": False,
                    "input_score_formula": "plm_prob * mut_prob^alpha",
                    "site_logistic_auroc": 0.8,
                    "site_logistic_pr_auc": 0.7,
                }
            ]
        )
        epoch_summary_df = pd.DataFrame(
            [
                {"model": "toy", "epoch_label": "raw_model", "epoch_value": 0.0, "n_lineages": 1}
            ]
        )
        panel_metadata = pd.DataFrame(
            [
                {"lineage": "lin1", "n_sequences": 2, "diversity_fasta": "diversity.fasta", "reference_fasta": "reference.fa"}
            ]
        )
        combined_df.to_csv(tables_dir / "combined_long_table.csv", index=False)
        alpha_df.to_csv(tables_dir / "alpha_sweep_fit_metrics.tsv", sep="\t", index=False)
        epoch_summary_df.to_csv(tables_dir / "epoch_metric_summary.tsv", sep="\t", index=False)
        panel_metadata.to_csv(tables_dir / "panel_metadata.tsv", sep="\t", index=False)

        captured = {}

        def fake_export_plots(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(rma, "export_plots", fake_export_plots)
        monkeypatch.setattr(rma, "build_lineage_cache", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not rebuild lineage cache")))

        args = _make_args(
            output_dir=output_dir,
            analysis_mode=None,
            mutation_model=None,
            model_tag=None,
            base_model=None,
            model_layer=None,
            checkpoint_dir=None,
            regen_figures_only=True,
        )

        result = rma.run_analysis(args)

        assert result == 0
        assert captured["output_dir"] == plots_dir
        pd.testing.assert_frame_equal(captured["combined_df"], combined_df)
        assert list(captured["alpha_df"].columns) == list(alpha_df.columns)
        assert float(captured["alpha_df"].iloc[0]["site_logistic_auroc"]) == pytest.approx(0.8)
        assert float(captured["alpha_df"].iloc[0]["site_logistic_pr_auc"]) == pytest.approx(0.7)

    def test_run_analysis_writes_expected_outputs(self, tmp_path, monkeypatch):
        output_dir = tmp_path / "out"
        args = _make_args(output_dir=output_dir, checkpoint_dir=tmp_path / "checkpoints")

        lineage_cache = {
            "lin1": {
                "lineage_key": "lin1",
                "records": [object(), object()],
                "full_ref_protein": "AA",
                "plm_ref_protein": "AA",
                "coord_map": {0: 0, 1: 1},
                "mut_profile": pd.DataFrame({1: {"A": 0.0, "C": 0.4}, 2: {"A": 0.0, "C": 0.2}}),
                "obs_freq": pd.DataFrame({1: {"A": 0.0, "C": 0.4}, 2: {"A": 0.0, "C": 0.1}}),
                "obs_depth": {1: 10, 2: 10},
                "alignment_diff_stats": {
                    "mapped_sites": 2,
                    "compared_sites": 2,
                    "differing_sites": 1,
                    "fixed_differing_sites": 0,
                },
                "diversity_path": "diversity.fasta",
                "reference_path": "reference.fa",
                "matched_pairs": 2,
                "any_nucleotide_diversity": False,
            }
        }
        monkeypatch.setattr(rma, "build_lineage_cache", lambda *a, **k: lineage_cache)
        monkeypatch.setattr(rma, "export_plots", lambda **kwargs: None)

        checkpoint_root = args.checkpoint_dir
        for name in ["checkpoint-10", "final_checkpoint"]:
            path = checkpoint_root / name
            path.mkdir(parents=True)
            (path / "model.safetensors").write_text(name)

        def fake_build_codon_aa_mutation_tables(model_name):
            return {"mutation_model_name": model_name}

        def fake_evaluate_alpha_sweep(df, alpha_grid, **kwargs):
            return pd.DataFrame(
                {
                    "alpha": [-1.0, 0.0],
                    "site_top10pct_mutated_enrichment": [0.1, 0.2],
                    "site_top10pct_mutated_precision": [0.1, 0.2],
                    "site_rank_spearman_r": [0.1, 0.2],
                    "mut_flat_global_spearman_r": [0.2, 0.3],
                    "mut_flat_global_pearson_r": [0.2, 0.3],
                    "mut_flat_mean_site_nll": [1.0, 0.8],
                }
            )

        _install_fake_functions_hf(
            monkeypatch,
            build_codon_aa_mutation_tables=fake_build_codon_aa_mutation_tables,
            evaluate_alpha_sweep=fake_evaluate_alpha_sweep,
        )

        monkeypatch.setattr(
            rma,
            "ensure_plm_matrix",
            lambda *a, **k: (
                pd.DataFrame([[0.0, 0.0], [0.5, 0.3]], index=["A", "C"], columns=[1, 2]),
                "plm.csv",
            ),
        )

        result = rma.run_analysis(args)

        assert result == 0
        assert (output_dir / "run_manifest.json").exists()
        assert (output_dir / "tables" / "combined_long_table.csv").exists()
        assert (output_dir / "tables" / "epoch_metric_summary.tsv").exists()
        assert (output_dir / "tables" / "best_alpha_two_methods.tsv").exists()
        assert (output_dir / "tables" / "alpha_sweep_fit_metrics_BY_LINEAGE.tsv").exists()
        assert (output_dir / "tables" / "per_model" / "ESMC_600M_FLU_raw_mutation_baseline_summary.tsv").exists()
        alpha_table = pd.read_csv(output_dir / "tables" / "per_model" / "ESMC_600M_FLU_raw_alpha_sweep_fit_metrics.tsv", sep="\t")
        alpha_by_lineage_table = pd.read_csv(output_dir / "tables" / "per_model" / "ESMC_600M_FLU_raw_alpha_sweep_fit_metrics_BY_LINEAGE.tsv", sep="\t")
        assert "model_variant" in alpha_table.columns
        assert "mutation_accessibility_only" in set(alpha_table["model_variant"])
        assert "n_lineages_averaged" in alpha_table.columns
        assert "site_logistic_auroc" in alpha_table.columns
        assert "lineage" in alpha_by_lineage_table.columns

    def test_export_plots_writes_latest_checkpoint_focused_plot_with_raw(self, tmp_path, monkeypatch):
        def fake_evaluate_alpha_sweep(df, alpha_grid, **kwargs):
            return pd.DataFrame(
                {
                    "alpha": [0.0, 1.0],
                    "mut_flat_global_spearman_r": [0.25, 0.3],
                    "mut_flat_nonzero_pearson_r": [0.2, 0.25],
                }
            )

        _install_fake_functions_hf(monkeypatch, evaluate_alpha_sweep=fake_evaluate_alpha_sweep)

        combined_df = pd.DataFrame(
            [
                {"model": "ESMC_600M_FLU_raw", "epoch_label": "raw_model", "epoch_value": 0.0, "lineage": "lin1", "position": 1, "ref_aa": "A", "aa": "C", "plm_prob": 0.2, "mut_prob": 0.4, "obs_freq": 0.4, "obs_present": 1, "depth": 10.0},
                {"model": "ESMC_600M_FLU_raw", "epoch_label": "raw_model", "epoch_value": 0.0, "lineage": "lin1", "position": 2, "ref_aa": "A", "aa": "C", "plm_prob": 0.1, "mut_prob": 0.2, "obs_freq": 0.0, "obs_present": 0, "depth": 10.0},
                {"model": "ESMC_600M_FLU_final_checkpoint", "epoch_label": "final_checkpoint", "epoch_value": 1.0, "lineage": "lin1", "position": 1, "ref_aa": "A", "aa": "C", "plm_prob": 0.3, "mut_prob": 0.4, "obs_freq": 0.4, "obs_present": 1, "depth": 10.0},
                {"model": "ESMC_600M_FLU_final_checkpoint", "epoch_label": "final_checkpoint", "epoch_value": 1.0, "lineage": "lin1", "position": 2, "ref_aa": "A", "aa": "C", "plm_prob": 0.2, "mut_prob": 0.2, "obs_freq": 0.0, "obs_present": 0, "depth": 10.0},
            ]
        )
        alpha_df = pd.DataFrame(
            [
                {
                    "alpha": 0.0,
                    "site_top10pct_mutated_enrichment": 0.2,
                    "site_top10pct_mutated_precision": 0.2,
                    "site_rank_spearman_r": 0.1,
                    "mut_flat_global_spearman_r": 0.3,
                    "mut_flat_global_pearson_r": 0.3,
                    "mut_flat_mean_site_nll": 0.8,
                    "mut_flat_nonzero_spearman_r": 0.3,
                    "mut_flat_nonzero_pearson_r": 0.3,
                    "mut_flat_logfreq_global_pearson_r": 0.3,
                    "mut_flat_logfreq_nonzero_pearson_r": 0.3,
                    "model": "ESMC_600M_FLU_raw",
                    "epoch_label": "raw_model",
                    "epoch_value": 0.0,
                },
                {
                    "alpha": 0.0,
                    "site_top10pct_mutated_enrichment": 0.4,
                    "site_top10pct_mutated_precision": 0.4,
                    "site_rank_spearman_r": 0.2,
                    "mut_flat_global_spearman_r": 0.5,
                    "mut_flat_global_pearson_r": 0.5,
                    "mut_flat_mean_site_nll": 0.6,
                    "mut_flat_nonzero_spearman_r": 0.5,
                    "mut_flat_nonzero_pearson_r": 0.5,
                    "mut_flat_logfreq_global_pearson_r": 0.5,
                    "mut_flat_logfreq_nonzero_pearson_r": 0.5,
                    "model": "ESMC_600M_FLU_final_checkpoint",
                    "epoch_label": "final_checkpoint",
                    "epoch_value": 1.0,
                },
            ]
        )

        rma.export_plots(
            output_dir=tmp_path,
            combined_df=combined_df,
            alpha_df=alpha_df,
            epoch_summary_df=pd.DataFrame(),
            scatter_alphas=[],
            scatter_max_points=100,
            lineage_cache={"lin1": {"n_sequences": 2}},
            dynamic_pseudocount=1e-3,
            mutation_baseline_x=-2.0,
            metrics_output_dir=tmp_path / "metrics",
        )

        latest_plot_dir = tmp_path / "per_model" / "ESMC_600M_FLU_final_checkpoint"
        assert (latest_plot_dir / "alpha_sweep_metrics_selected.png").exists()
        assert (latest_plot_dir / "alpha_sweep_metrics_selected_mutation_counts.png").exists()
        assert (latest_plot_dir / "alpha_sweep_logistic_metrics_selected.png").exists()
        assert (latest_plot_dir / "alpha_sweep_logistic_metrics_selected_mutation_counts.png").exists()
        assert (latest_plot_dir / "alpha_sweep_metrics_selected_with_raw.png").exists()
        assert (latest_plot_dir / "alpha_sweep_metrics_selected_with_raw_mutation_counts.png").exists()
        assert (tmp_path / "alpha_sweep_logistic_metrics_selected.png").exists()
        assert (tmp_path / "alpha_sweep_logistic_metrics_selected_mutation_counts.png").exists()
        assert (tmp_path / "alpha_sweep_metrics_selected_mutation_counts.png").exists()
        assert (tmp_path / "hurdle_alpha_sweep.png").exists()
        assert (tmp_path / "hurdle_regression_diagnostics.png").exists()
        assert (latest_plot_dir / "hurdle_alpha_sweep.png").exists()
        assert (latest_plot_dir / "hurdle_regression_diagnostics.png").exists()
        assert (tmp_path / "metrics" / "hurdle_alpha_sweep_metrics.csv").exists()
        assert (tmp_path / "metrics" / "hurdle_model_summary.csv").exists()
        assert (tmp_path / "metrics" / "logistic_regression_comparison_report.tsv").exists()
        assert (tmp_path / "metrics" / "logistic_regression_comparison_notes.txt").exists()
        assert (tmp_path / "metrics" / "per_model" / "ESMC_600M_FLU_final_checkpoint" / "hurdle_alpha_sweep_metrics.csv").exists()
        assert (tmp_path / "metrics" / "per_model" / "ESMC_600M_FLU_final_checkpoint" / "hurdle_model_summary.csv").exists()
        assert (tmp_path / "metrics" / "per_model" / "ESMC_600M_FLU_final_checkpoint" / "logistic_regression_comparison_report.tsv").exists()

        hurdle_summary = pd.read_csv(tmp_path / "metrics" / "hurdle_model_summary.csv")
        assert {"mutation_only", "plm_only_alpha0", "best_alpha_hurdle", "two_input_hurdle"}.issubset(set(hurdle_summary["model_variant"]))
        baseline_rows = hurdle_summary.loc[hurdle_summary["model_variant"] == "mutation_only"]
        assert baseline_rows["alpha_presence"].eq(1.1).all()
        assert baseline_rows["alpha_frequency"].eq(1.1).all()
        plm_only_rows = hurdle_summary.loc[hurdle_summary["model_variant"] == "plm_only_alpha0"]
        assert plm_only_rows["alpha_presence"].eq(0.0).all()
        assert plm_only_rows["alpha_frequency"].eq(0.0).all()
        hurdle_points = pd.read_csv(tmp_path / "metrics" / "hurdle_alpha_sweep_metrics.csv")
        assert {"ESMC_600M_FLU_raw", "ESMC_600M_FLU_final_checkpoint"}.issubset(set(hurdle_points["model"]))
        assert {"presence_score_formula", "frequency_score_formula", "freq_response_definition", "freq_raw_response_definition", "freq_raw_r2"}.issubset(hurdle_points.columns)
        assert hurdle_points["presence_score_formula"].eq("log10(plm_prob) + alpha_presence * log10(mut_prob)").all()
        logistic_report = pd.read_csv(tmp_path / "metrics" / "logistic_regression_comparison_report.tsv", sep="\t")
        assert {"standalone_binary_term", "hurdle_binary_term"}.issubset(set(logistic_report["framework"]))
        assert {"plm_prob", "mutation_accessibility"}.issubset(set(logistic_report["predictor"]))

    def test_hurdle_alpha_sweep_summary_includes_expected_models(self):
        combined_df = pd.DataFrame(
            [
                {"lineage": "lin1", "position": 1, "ref_aa": "A", "aa": "C", "plm_prob": 0.80, "mut_prob": 0.70, "obs_freq": 0.30, "obs_present": 1},
                {"lineage": "lin1", "position": 1, "ref_aa": "A", "aa": "G", "plm_prob": 0.20, "mut_prob": 0.20, "obs_freq": 0.00, "obs_present": 0},
                {"lineage": "lin1", "position": 2, "ref_aa": "T", "aa": "C", "plm_prob": 0.60, "mut_prob": 0.50, "obs_freq": 0.15, "obs_present": 1},
                {"lineage": "lin1", "position": 2, "ref_aa": "T", "aa": "G", "plm_prob": 0.10, "mut_prob": 0.15, "obs_freq": 0.00, "obs_present": 0},
                {"lineage": "lin2", "position": 1, "ref_aa": "A", "aa": "C", "plm_prob": 0.75, "mut_prob": 0.65, "obs_freq": 0.25, "obs_present": 1},
                {"lineage": "lin2", "position": 1, "ref_aa": "A", "aa": "G", "plm_prob": 0.25, "mut_prob": 0.25, "obs_freq": 0.00, "obs_present": 0},
                {"lineage": "lin2", "position": 2, "ref_aa": "T", "aa": "C", "plm_prob": 0.55, "mut_prob": 0.45, "obs_freq": 0.12, "obs_present": 1},
                {"lineage": "lin2", "position": 2, "ref_aa": "T", "aa": "G", "plm_prob": 0.15, "mut_prob": 0.10, "obs_freq": 0.00, "obs_present": 0},
            ]
        )

        hurdle_alpha_df = rma.evaluate_hurdle_alpha_sweep(combined_df, [0.0, 1.0])
        hurdle_summary_df = rma.summarize_hurdle_models(combined_df, hurdle_alpha_df)

        assert len(hurdle_alpha_df) == 4
        assert {"alpha_presence", "alpha_frequency", "logistic_tjur_r2", "freq_log10_r2", "freq_raw_r2", "hurdle_mean_r2", "hurdle_mean_r2_raw_frequency"}.issubset(hurdle_alpha_df.columns)
        assert hurdle_alpha_df["presence_score_formula"].eq("log10(plm_prob) + alpha_presence * log10(mut_prob)").all()
        assert {"mutation_only", "plm_only_alpha0", "best_alpha_hurdle", "two_input_hurdle"}.issubset(set(hurdle_summary_df["model_variant"]))
        assert {"logistic_intercept", "freq_intercept", "freq_raw_intercept", "hurdle_mean_r2", "hurdle_mean_r2_raw_frequency"}.issubset(hurdle_summary_df.columns)
        mutation_only_row = hurdle_summary_df.loc[hurdle_summary_df["model_variant"] == "mutation_only"].iloc[0]
        assert float(mutation_only_row["alpha_presence"]) == 1.1
        assert float(mutation_only_row["alpha_frequency"]) == 1.1
        plm_only_row = hurdle_summary_df.loc[hurdle_summary_df["model_variant"] == "plm_only_alpha0"].iloc[0]
        assert float(plm_only_row["alpha_presence"]) == 0.0
        assert float(plm_only_row["alpha_frequency"]) == 0.0

    def test_run_analysis_uses_cached_per_model_tables_without_rebuilding_lineages(self, tmp_path, monkeypatch):
        output_dir = tmp_path / "out"
        model_tables_dir = output_dir / "tables" / "per_model"
        model_tables_dir.mkdir(parents=True)
        checkpoint_root = tmp_path / "checkpoints"
        checkpoint_dir = checkpoint_root / "checkpoint-10"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "model.safetensors").write_text("ckpt-10")

        args = _make_args(output_dir=output_dir, checkpoint_dir=checkpoint_root)

        panel_metadata = pd.DataFrame(
            [
                {
                    "model": "ESMC_600M_FLU_raw",
                    "epoch_label": "raw_model",
                    "epoch_value": 0.0,
                    "lineage": "lin1",
                    "n_sequences": 2,
                    "reference_length": 2,
                    "mapped_ref_sites": 2,
                    "compared_sites_non_gap_non_stop": 2,
                    "differing_sites_vs_reference_non_gap_non_stop": 1,
                    "fixed_differing_sites_vs_reference_non_gap_non_stop": 0,
                    "diversity_fasta": "diversity.fasta",
                    "reference_fasta": "reference.fa",
                    "plm_profile": "plm_raw.csv",
                    "diversity_sequences_detected_as_nucleotide": False,
                },
                {
                    "model": "ESMC_600M_FLU_checkpoint-10",
                    "epoch_label": "checkpoint-10",
                    "epoch_value": 1.0,
                    "lineage": "lin1",
                    "n_sequences": 2,
                    "reference_length": 2,
                    "mapped_ref_sites": 2,
                    "compared_sites_non_gap_non_stop": 2,
                    "differing_sites_vs_reference_non_gap_non_stop": 1,
                    "fixed_differing_sites_vs_reference_non_gap_non_stop": 0,
                    "diversity_fasta": "diversity.fasta",
                    "reference_fasta": "reference.fa",
                    "plm_profile": "plm_ckpt.csv",
                    "diversity_sequences_detected_as_nucleotide": False,
                },
            ]
        )
        (output_dir / "tables").mkdir(parents=True, exist_ok=True)
        panel_metadata.to_csv(output_dir / "tables" / "panel_metadata.tsv", sep="\t", index=False)

        combined_df = pd.DataFrame(
            [
                {"model": "ESMC_600M_FLU_raw", "epoch_label": "raw_model", "epoch_value": 0.0, "lineage": "lin1", "position": 1, "ref_aa": "A", "aa": "C", "plm_prob": 0.90, "mut_prob": 0.80, "obs_freq": 0.40, "obs_present": 1, "depth": 10.0},
                {"model": "ESMC_600M_FLU_raw", "epoch_label": "raw_model", "epoch_value": 0.0, "lineage": "lin1", "position": 1, "ref_aa": "A", "aa": "G", "plm_prob": 0.70, "mut_prob": 0.60, "obs_freq": 0.25, "obs_present": 1, "depth": 10.0},
                {"model": "ESMC_600M_FLU_raw", "epoch_label": "raw_model", "epoch_value": 0.0, "lineage": "lin1", "position": 2, "ref_aa": "A", "aa": "C", "plm_prob": 0.20, "mut_prob": 0.30, "obs_freq": 0.00, "obs_present": 0, "depth": 10.0},
                {"model": "ESMC_600M_FLU_raw", "epoch_label": "raw_model", "epoch_value": 0.0, "lineage": "lin1", "position": 2, "ref_aa": "A", "aa": "G", "plm_prob": 0.10, "mut_prob": 0.20, "obs_freq": 0.00, "obs_present": 0, "depth": 10.0},
            ]
        )
        alpha_df = pd.DataFrame(
            [
                {
                    "alpha": 0.0,
                    "site_top10pct_mutated_enrichment": 0.2,
                    "site_top10pct_mutated_precision": 0.2,
                    "site_rank_spearman_r": 0.1,
                    "mut_flat_global_spearman_r": 0.3,
                    "mut_flat_global_pearson_r": 0.3,
                    "mut_flat_mean_site_nll": 0.8,
                    "model": "ESMC_600M_FLU_raw",
                    "epoch_label": "raw_model",
                    "epoch_value": 0.0,
                }
            ]
        )
        combined_df.to_csv(model_tables_dir / "ESMC_600M_FLU_raw_combined_long_table.csv", index=False)
        alpha_df.to_csv(model_tables_dir / "ESMC_600M_FLU_raw_alpha_sweep_fit_metrics.tsv", sep="\t", index=False)
        alpha_df.assign(lineage="lin1").to_csv(model_tables_dir / "ESMC_600M_FLU_raw_alpha_sweep_fit_metrics_BY_LINEAGE.tsv", sep="\t", index=False)

        combined_df_ckpt = combined_df.copy()
        combined_df_ckpt["model"] = "ESMC_600M_FLU_checkpoint-10"
        combined_df_ckpt["epoch_label"] = "checkpoint-10"
        combined_df_ckpt["epoch_value"] = 1.0
        alpha_df_ckpt = alpha_df.copy()
        alpha_df_ckpt["model"] = "ESMC_600M_FLU_checkpoint-10"
        alpha_df_ckpt["epoch_label"] = "checkpoint-10"
        alpha_df_ckpt["epoch_value"] = 1.0
        combined_df_ckpt.to_csv(model_tables_dir / "ESMC_600M_FLU_checkpoint-10_combined_long_table.csv", index=False)
        alpha_df_ckpt.to_csv(model_tables_dir / "ESMC_600M_FLU_checkpoint-10_alpha_sweep_fit_metrics.tsv", sep="\t", index=False)
        alpha_df_ckpt.assign(lineage="lin1").to_csv(model_tables_dir / "ESMC_600M_FLU_checkpoint-10_alpha_sweep_fit_metrics_BY_LINEAGE.tsv", sep="\t", index=False)

        monkeypatch.setattr(rma, "build_lineage_cache", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not rebuild lineage cache")))
        monkeypatch.setattr(rma, "export_plots", lambda **kwargs: None)

        result = rma.run_analysis(args)

        assert result == 0
        alpha_table = pd.read_csv(output_dir / "tables" / "alpha_sweep_fit_metrics.tsv", sep="\t")
        assert alpha_table["site_logistic_auroc"].notna().all()
        assert alpha_table["site_logistic_pr_auc"].notna().all()
        status_df = pd.read_csv(output_dir / "tables" / "model_run_status.tsv", sep="\t")
        cached_rows = status_df.loc[
            (status_df["lineage"] == "all")
            & (status_df["model"].isin(["ESMC_600M_FLU_raw", "ESMC_600M_FLU_checkpoint-10"]))
            & (status_df["reason"] == "cached")
        ]
        assert set(cached_rows["model"]) == {"ESMC_600M_FLU_raw", "ESMC_600M_FLU_checkpoint-10"}


class TestManifest:
    def test_save_run_manifest_uses_filter_fixed_mutations_key(self, tmp_path):
        args = _make_args(output_dir=tmp_path, checkpoint_dir=tmp_path / "checkpoint-root")
        rma.save_run_manifest(args, tmp_path, [{"label": "lin1", "diversity_path": "d.fa", "reference_path": "r.fa"}])

        manifest = pd.read_json(tmp_path / "run_manifest.json", typ="series")
        assert "filter_fixed_mutations" in manifest.index