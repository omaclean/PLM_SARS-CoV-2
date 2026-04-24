from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_lineage_subalignments_general_SC2.py"
SPEC = spec_from_file_location("build_lineage_subalignments_general_SC2", MODULE_PATH)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_hamming_distance_ignores_unknown_residues() -> None:
    assert MODULE.hamming_distance("AXCDFN", "ABCDYQ") == 1


def test_branch_mutations_skip_unknown_reference_sites() -> None:
    assert MODULE.branch_mutations("ABXD", "ACND") == [(1, "B", "C")]


def test_mode_criteria_do_not_fail_on_unknown_current_branch_residue() -> None:
    branch_by_ref = [[], [(1, "B", "C")], [(2, "D", "E")]]
    cumulative_by_ref = [[], [(1, "B", "C")], [(1, "B", "C"), (2, "D", "E")]]

    passes, next_present, next_total = MODULE.passes_mode_criteria(
        query_seq="AXD",
        ref_index=1,
        branch_by_ref=branch_by_ref,
        cumulative_by_ref=cumulative_by_ref,
        mode="hard",
        hard_max_next=0,
        final_lineage_max_missing_current_branch=0,
    )

    assert passes is True
    assert next_present == 0
    assert next_total == 1


def test_mode_criteria_still_counts_real_next_branch_matches() -> None:
    branch_by_ref = [[], [(1, "B", "C")], [(2, "D", "E")]]
    cumulative_by_ref = [[], [(1, "B", "C")], [(1, "B", "C"), (2, "D", "E")]]

    passes, next_present, next_total = MODULE.passes_mode_criteria(
        query_seq="ACE",
        ref_index=1,
        branch_by_ref=branch_by_ref,
        cumulative_by_ref=cumulative_by_ref,
        mode="hard",
        hard_max_next=0,
        final_lineage_max_missing_current_branch=0,
    )

    assert passes is False
    assert next_present == 1
    assert next_total == 1