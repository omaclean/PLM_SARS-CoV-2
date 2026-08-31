#!/usr/bin/env python3
"""Validation of the SCAFFOLDING, not of the pipeline.

Every other module in this suite tests the pipeline.  This one tests the
fixtures the others stand on, and it exists because a fixture that quietly stops
being what its docstring claims takes a whole suite down with it -- silently,
because the tests keep passing against the new, wrong ground truth.

Each test here answers one question: *is the baked-in expected value still the
value the fixture actually produces, and is it still the value the real module
consumes?*  So:

* the literal parent map, trace defaults and row orders in ``conftest`` still
  match ``prescott_iav.constants`` (they are literals precisely so a regression
  to the old ``K <- J.2_int`` edge cannot hide);
* the synthetic CDS still translates to the synthetic protein;
* the hand-worked Henikoff weights are still what the implementation computes;
* the CA-ladder circular variances are still exactly 0.0 and 1.0;
* the R ``write.table`` fixture still round-trips through the real parser;
* the planted panel counts still produce exactly the expected frequency file.

SIBLING AGENTS: please do not extend this file with pipeline tests -- put those
in your own module.  Add to this one only when you add a fixture whose ground
truth needs pinning.
"""

from __future__ import annotations

import numpy as np
import pytest

from prescott_iav import common, constants, jet_surrogate, prepare_inputs, run_escott
from tests_prescott_iav import conftest as C

pytestmark = pytest.mark.unit


class TestPinnedConstants:
    """The literals in conftest are the independent copy; these keep them honest."""

    def test_parent_map_is_the_corrected_ladder(self):
        assert constants.DEFAULT_PARENT_MAPS["clade_evidence"] == C.EXPECTED_PARENT_MAP
        assert constants.DEFAULT_PARENT_MAP_PRESET == "clade_evidence"
        # The contested edge, spelled out: K descends from J.2.4, not J.2_int.
        child, default_parent, sensitivity_parent = C.CONTESTED_EDGE
        assert constants.DEFAULT_PARENT_MAPS["clade_evidence"][child] == default_parent
        assert constants.DEFAULT_PARENT_MAPS["brief_as_stated"][child] == sensitivity_parent

    def test_brief_as_stated_is_only_the_sensitivity_alternative(self):
        assert constants.DEFAULT_PARENT_MAPS["brief_as_stated"] == C.EXPECTED_SENSITIVITY_PARENT_MAP
        assert constants.DEFAULT_PARENT_MAP_PRESET != "brief_as_stated"

    def test_topology_metadata(self):
        assert set(constants.INPUT_ONLY_LINEAGES) == set(C.EXPECTED_INPUT_ONLY_LINEAGES)
        assert constants.LINEAGE_TAGS == C.EXPECTED_LINEAGE_TAGS

    def test_trace_defaults(self):
        assert constants.DEFAULT_TRACE_TOP_FRACTION == C.EXPECTED_TRACE_TOP_FRACTION
        assert constants.MAX_ZERO_TRACE_FRACTION == C.EXPECTED_MAX_ZERO_TRACE_FRACTION
        assert constants.WARN_ZERO_TRACE_FRACTION == C.EXPECTED_WARN_ZERO_TRACE_FRACTION

    def test_row_orders_and_sentinels(self):
        assert run_escott.PLM_CACHE_AA_ORDER == C.PLM_CACHE_ROW_ORDER
        assert run_escott.ESCOTT_AA_ORDER == C.ESCOTT_ROW_ORDER
        assert run_escott.NO_FREQUENCY_SENTINEL == 999.0
        assert run_escott.SUPPORTED_PRESCOTT_EQUATIONS == (1, 2, 3, 5)


class TestQuerySequences:
    def test_cds_translates_to_the_protein(self, query_cds_fasta):
        loaded = common.load_reference_cds(query_cds_fasta, "K")
        assert loaded["protein"] == C.QUERY_PROTEIN
        assert len(loaded["nucleotide"]) == 3 * C.QUERY_LENGTH + 3

    def test_header_yields_the_expected_escott_token(self, query_protein_fasta):
        assert common.escott_prot_token(C.QUERY_HEADER) == "HAK"
        assert run_escott.escott_prot_token(query_protein_fasta) == "HAK"

    def test_parent_differs_at_exactly_one_position(self):
        differences = [
            i + 1 for i, (a, b) in enumerate(zip(C.QUERY_PROTEIN, C.PARENT_PROTEIN)) if a != b
        ]
        assert differences == [40]


class TestAlignments:
    def test_uniform_msa_weights_are_exactly_one(self, uniform_msa):
        encoded = jet_surrogate.encode_msa(C.read_fasta(uniform_msa["path"]))
        weights = jet_surrogate.henikoff_weights(encoded)
        assert np.allclose(weights, uniform_msa["expected_weights"], atol=1e-12)
        _kl, occupancy = jet_surrogate.column_conservation(encoded, weights)
        assert np.allclose(occupancy, uniform_msa["expected_occupancy"], atol=1e-12)

    def test_handworked_weights_match_the_arithmetic_in_the_docstring(self, handworked_msa):
        encoded = jet_surrogate.encode_msa(C.read_fasta(handworked_msa["path"]))
        weights = jet_surrogate.henikoff_weights(encoded)
        assert np.allclose(weights, handworked_msa["expected_weights"], atol=1e-12), weights

    def test_tiny_msa_column_classes_behave_as_advertised(self, tiny_msa):
        encoded = jet_surrogate.encode_msa(C.read_fasta(tiny_msa["path"]))
        assert encoded.shape == (tiny_msa["n_rows"], tiny_msa["n_columns"])
        weights = jet_surrogate.henikoff_weights(encoded)
        kl_bits, occupancy = jet_surrogate.column_conservation(encoded, weights)

        for pos in tiny_msa["conserved_positions"]:
            assert occupancy[pos - 1] == pytest.approx(1.0, abs=1e-12)
        for pos in tiny_msa["all_gap_positions"]:
            # Only the query has a residue, so occupancy is one row's weight share.
            assert occupancy[pos - 1] < 0.2
        # Conservation must order the classes the way the fixture claims.
        for conserved in tiny_msa["conserved_positions"]:
            for hypervariable in tiny_msa["hypervariable_positions"]:
                assert kl_bits[conserved - 1] > kl_bits[hypervariable - 1]

    def test_gapped_query_msa_is_refused(self, gapped_query_msa, query_protein_fasta):
        with pytest.raises(ValueError, match="gap"):
            jet_surrogate.build_jet_table(gapped_query_msa, query_protein_fasta, None, None)


@pytest.mark.requires_prody
@pytest.mark.requires_scipy
class TestStructureGeometry:
    def test_ladder_circular_variance_is_exactly_zero_or_one(self, cv_ladder_pdb):
        struct = jet_surrogate.load_structure(cv_ladder_pdb["path"])
        got = jet_surrogate.circular_variance(
            struct, "A", range(1, cv_ladder_pdb["n_residues"] + 1), radius=7.0
        )
        assert got == pytest.approx(cv_ladder_pdb["expected_cv"], abs=1e-9), got

    def test_context_chain_buries_the_terminal_residue(self, cv_context_pdb):
        struct = jet_surrogate.load_structure(cv_context_pdb["path"])
        got = jet_surrogate.circular_variance(struct, "A", range(1, 9), radius=7.0)
        assert got == pytest.approx(cv_context_pdb["expected_cv"], abs=1e-9), got
        # The whole point: only the interface residue moved.
        assert got[cv_context_pdb["changed_position"]] == pytest.approx(1.0, abs=1e-9)

    def test_query_numbered_pdb_covers_what_it_says(self, query_numbered_pdb_factory):
        partial = query_numbered_pdb_factory(covered=range(1, 51))
        index = jet_surrogate.residue_index(jet_surrogate.load_structure(partial), "A")
        assert sorted(index) == list(range(1, 51))

        trimer = query_numbered_pdb_factory(chains=("A", "B", "C"))
        struct = jet_surrogate.load_structure(trimer)
        for chain in "ABC":
            assert sorted(jet_surrogate.residue_index(struct, chain)) == list(
                range(1, C.QUERY_LENGTH + 1)
            )


@pytest.mark.requires_freesasa
class TestStructureSasa:
    def test_isolated_glycine_clips_to_one(self, sasa_monomer_pdb):
        rsa = jet_surrogate.relative_sasa(sasa_monomer_pdb["path"], "A")
        assert rsa == pytest.approx(sasa_monomer_pdb["expected_rsa"])

    def test_shell_buries_the_central_residue(self, sasa_context_pdb):
        rsa = jet_surrogate.relative_sasa(sasa_context_pdb["path"], "A")
        assert rsa[sasa_context_pdb["buried_resnum"]] <= sasa_context_pdb["expected_buried_max_rsa"]
        assert rsa[sasa_context_pdb["isolated_resnum"]] == sasa_context_pdb["expected_isolated_rsa"]


class TestPipelineArtefacts:
    def test_jet_res_parses_and_validates(self, fake_jet_res):
        table = jet_surrogate.read_jet_res(fake_jet_res["path"])
        assert list(table.columns) == fake_jet_res["columns"]
        jet_surrogate.validate_jet_table(table, expected_rows=fake_jet_res["n_rows"])
        n_zero = int((table["trace"].to_numpy() == 0.0).sum())
        assert n_zero == fake_jet_res["n_zero_trace"]
        frac = n_zero / fake_jet_res["n_rows"]
        assert frac == pytest.approx(fake_jet_res["expected_frac_zero"])
        # Deliberately in the warn band: warn, but do not refuse.
        assert fake_jet_res["warn_fraction"] < frac < fake_jet_res["max_fraction"]

    def test_jet_res_factory_straddles_the_refusal_ceiling(self, jet_res_factory):
        _low_path, low = jet_res_factory(n_zero=7)
        _high_path, high = jet_res_factory(n_zero=30)
        assert low["frac_zero"] < C.EXPECTED_MAX_ZERO_TRACE_FRACTION < high["frac_zero"]

    def test_escott_matrix_round_trips_through_the_real_parser(self, fake_escott_matrix):
        frame = run_escott.read_escott_matrix(fake_escott_matrix["path"])
        assert frame.shape == (20, fake_escott_matrix["n_positions"])
        assert run_escott.escott_wt_sequence(frame) == fake_escott_matrix["wt_sequence"]
        for (aa, position), expected in fake_escott_matrix["values"].items():
            observed = frame.at[aa, position]
            if expected is None:
                assert np.isnan(observed), (aa, position)
            else:
                assert observed == pytest.approx(expected, abs=1e-12), (aa, position)

    def test_flat_columns_softmax_to_exactly_one_twentieth(self, fake_escott_matrix):
        frame = run_escott.read_escott_matrix(fake_escott_matrix["path"])
        probabilities = run_escott.escott_to_probability(frame, temperature=1.0)
        for position in fake_escott_matrix["flat_positions"]:
            assert np.allclose(
                probabilities[position].to_numpy(),
                fake_escott_matrix["expected_flat_probability"],
                atol=1e-12,
            ), position
        assert run_escott.count_flat_columns(frame) == len(fake_escott_matrix["flat_positions"])

    def test_score_matrix_layout(self, score_matrix_factory):
        import pandas as pd

        raw = pd.read_csv(score_matrix_factory(), index_col=0, header=None)
        assert raw.shape == (21, C.QUERY_LENGTH)
        assert "".join(raw.iloc[0].tolist()) == C.QUERY_PROTEIN
        assert list(raw.index[1:]) == list(C.PLM_CACHE_ROW_ORDER)

    def test_frequency_file_is_read_by_the_real_loader(self, frequency_file_factory):
        loaded = run_escott.load_frequency_file(frequency_file_factory())
        assert loaded == pytest.approx(C.EXPECTED_FREQUENCY_FILE_MIN_COUNT_1)


class TestPanels:
    @pytest.mark.parametrize("min_count", [1, 2])
    def test_planted_counts_produce_the_expected_frequency_file(
        self, frequency_panels, tmp_path, min_count
    ):
        expected = frequency_panels[f"expected_frequency_file_min_count_{min_count}"]
        out = tmp_path / f"freq_{min_count}.txt"
        report = prepare_inputs.build_parent_frequency_file(
            child_label="K",
            parent_label="J.2.4",
            child_protein=frequency_panels["child_protein"],
            parent_panel_fasta=frequency_panels["parent_fasta"],
            out_txt=out,
            out_meta=tmp_path / f"freq_{min_count}_meta.tsv",
            min_count=min_count,
            min_depth=50,
            freq_max=0.95,
            parent_protein=frequency_panels["parent_protein"],
            drop_parent_reversions=True,
        )
        assert run_escott.load_frequency_file(out) == pytest.approx(expected)
        assert report["mapped_ref_sites"] == C.QUERY_LENGTH
        assert report["median_mapped_depth"] == frequency_panels["expected_median_depth"]
        assert report["n_reverted_ancestral_mutants"] == frequency_panels[
            "expected_n_reverted_ancestral"
        ]
        assert report["n_parent_reversion_mutants"] == frequency_panels[
            "expected_n_parent_reversions"
        ]

    def test_target_panel_has_the_planted_composition(self, frequency_panels):
        counts = frequency_panels["target_counts"]
        depths = frequency_panels["target_depths"]
        assert counts[18]["K"] == 50 and counts[18]["R"] == 50 and depths[18] == 100
        assert counts[60]["H"] == 9 and depths[60] == 90
        assert frequency_panels["target_frequencies"]["Q60H"] == pytest.approx(0.10)

    def test_panel_factory_ground_truth_matches_the_file(self, panel_factory):
        path, truth = panel_factory({7: {"W": 25}})
        records = C.read_fasta(path)
        assert len(records) == C.PANEL_N_RECORDS
        assert sum(1 for _, seq in records if seq[6] == "W") == 25
        assert truth["frequencies"][f"{C.QUERY_PROTEIN[6]}7W"] == pytest.approx(0.25)


class TestLeakagePanels:
    def test_planted_duplicate_is_the_only_shared_sequence(self, leakage_panels):
        from prescott_iav import leakage_check as lc

        target = {lc.sequence_hash(seq) for _, seq in leakage_panels["target_records"]}
        parent = {lc.sequence_hash(seq) for _, seq in leakage_panels["parent_records"]}
        assert len(target & parent) == leakage_panels["expected_shared_exact_sequences"]

    def test_accessions_are_disjoint(self, leakage_panels):
        target = {header for header, _ in leakage_panels["target_records"]}
        parent = {header for header, _ in leakage_panels["parent_records"]}
        assert len(target & parent) == leakage_panels["expected_shared_accessions"]

    def test_deep_set_leak_is_invisible_to_hashing(self, leakage_panels):
        """The planted deep-set leak carries the signal peptide, so it does NOT hash
        to its panel twin.  That is the real geometry, and it is why only an
        alignment-based check can find it."""
        from prescott_iav import leakage_check as lc

        target = {lc.sequence_hash(seq) for _, seq in leakage_panels["target_records"]}
        deep = {lc.sequence_hash(seq) for _, seq in leakage_panels["deep_records"]}
        assert len(target & deep) == 0

        planted = leakage_panels["deep_records"][leakage_panels["planted_deep_row"]][1]
        twin = leakage_panels["target_records"][0][1]
        assert planted == leakage_panels["signal_peptide"] + twin


class TestPreparedTree:
    def test_run_escott_resolves_every_lineage(self, prepared_inputs_tree):
        for label in prepared_inputs_tree["evaluable"]:
            info = run_escott.resolve_lineage_inputs(prepared_inputs_tree["inputs_dir"], label)
            assert info["protein"] == C.QUERY_PROTEIN
            assert info["prot_token"] == C.EXPECTED_PROT_TOKENS[label]
            assert info["frequency_path"] is not None

    def test_alternate_parent_is_discoverable_by_label(self, prepared_inputs_tree):
        alternates = run_escott.resolve_alternate_frequency_paths(
            prepared_inputs_tree["inputs_dir"], "K"
        )
        assert set(alternates) == {"J.2_int"}
        assert alternates["J.2_int"]["path"].exists()

    def test_manifest_carries_the_corrected_parent_map(self, prepared_inputs_tree):
        assert run_escott.resolve_parent_map(
            prepared_inputs_tree["inputs_dir"]
        ) == C.EXPECTED_PARENT_MAP


class TestGuide:
    def test_guide_rows_and_parent_map_resolution(self, five_lineage_guide):
        rows = common.read_guide_rows(five_lineage_guide["path"])
        assert [row["label"] for row in rows] == list(C.LINEAGE_ORDER)
        resolved = common.resolve_parent_map(
            "clade_evidence", None, [row["label"] for row in rows]
        )
        assert resolved == C.EXPECTED_PARENT_MAP

    def test_output_dir_factory_creates_the_diagnostics_dir(self, output_dir_factory):
        root = output_dir_factory("run1")
        for sub in ("tables", "tables/diagnostics", "figures", "scores", "inputs"):
            assert (root / sub).is_dir(), sub
