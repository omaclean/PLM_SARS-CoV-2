# %%
import sys
sys.path.append('../../')

from Functions_HuggingFace import *

import re
import os
from Bio import PDB, Align
from Bio.SeqUtils import seq1
import py3Dmol


pdb_path = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/4WE4_assembly.pdb"
# pdb_path= "/home3/oml4h/PLM_SARS-CoV-2/Sequences/4O5N-assembly1.cif"
# pdb_path="/home3/oml4h/PLM_SARS-CoV-2/Sequences/4WE8-assembly1.cif"
pdb_path="/home3/oml4h/PLM_SARS-CoV-2/Sequences/6WXB-assembly1.cif"

pdb_path="/home3/oml4h/PLM_SARS-CoV-2/Sequences/viro3d_CF-CAA24272.1_9914_relaxed.pdb"
pdb_path="/home3/oml4h/PLM_SARS-CoV-2/Sequences/EPI4748783_HA_A_England_01837755_2025_EPI_ISL_20210731_J_2_4_1_model.cif"
pdb_path="/home3/oml4h/PLM_SARS-CoV-2/Sequences/Joe_new_K_lin.pdb"
# pdb_path="/home3/oml4h/PLM_SARS-CoV-2/Sequences/7ZJ6-assembly1.cif"
# pdb_path="/home3/oml4h/PLM_SARS-CoV-2/Sequences/7ZJ8-assembly1.cif"
pdb_path="/home3/oml4h/PLM_SARS-CoV-2/Sequences/7ZJ7-assembly1.cif"
membrane_pdb_path = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/membrane_dppc128.pdb"
pdb_path="/home3/oml4h/PLM_SARS-CoV-2/Sequences/EPI4748783_HA_A_England_01837755_2025_EPI_ISL_20210731_J_2_4_1_model.cif"
sequences = read_sequences_to_dict(
    "/home3/oml4h/PLM_SARS-CoV-2/Sequences/huH3N2_HA_CDS.translated_OM_synth_extra_steps.fas"
)

reference_path = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/H3N2_canonical.fa"

seq1_index=0
seq2_index=4
output_dir = "/home3/oml4h/PLM_SARS-CoV-2/Results/structure_play"

output_dir = "/home3/oml4h/PLM_SARS-CoV-2/Results/structure_playJ.2"
os.makedirs(output_dir, exist_ok=True)


def _extract_pdb_chain_sequences(pdb_file):
    if pdb_file.lower().endswith((".cif", ".mmcif")):
        parser = PDB.MMCIFParser(QUIET=True)
    else:
        parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("struct", pdb_file)
    chain_data = {}
    for model in structure:
        for chain in model:
            seq_chars = []
            residue_ids = []
            hetero_resnames = []
            for residue in chain:
                if PDB.is_aa(residue):
                    try:
                        aa = seq1(residue.get_resname())
                    except Exception:
                        aa = "X"
                    seq_chars.append(aa)
                    residue_ids.append(residue.get_id())
                else:
                    resname = residue.get_resname()
                    if resname and resname != "HOH":
                        hetero_resnames.append(resname)
            chain_data[chain.id] = {
                "seq": "".join(seq_chars),
                "residue_ids": residue_ids,
                "hetero_resnames": sorted(set(hetero_resnames)),
            }
    return chain_data


def _extract_plddt_by_chain(pdb_file):
    if pdb_file.lower().endswith((".cif", ".mmcif")):
        parser = PDB.MMCIFParser(QUIET=True)
    else:
        parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("struct", pdb_file)
    chain_plddt = {}

    for model in structure:
        for chain in model:
            plddt_values = []
            for residue in chain:
                if not PDB.is_aa(residue):
                    continue
                if "CA" in residue:
                    plddt_values.append(residue["CA"].get_bfactor())
                else:
                    atom_b = [atom.get_bfactor() for atom in residue.get_atoms()]
                    plddt_values.append(sum(atom_b) / len(atom_b) if atom_b else None)
            if plddt_values:
                chain_plddt[chain.id] = plddt_values
    return chain_plddt


def _alignment_indices(alignment):
    try:
        return alignment.indices[0], alignment.indices[1]
    except AttributeError:
        user_indices = []
        pdb_indices = []
        for (u_start, u_end), (p_start, p_end) in zip(*alignment.aligned):
            user_indices.extend(range(u_start, u_end))
            pdb_indices.extend(range(p_start, p_end))
        return user_indices, pdb_indices


def _format_res_id(res_id):
    resseq = res_id[1]
    icode = res_id[2].strip()
    return f"{resseq}{icode}" if icode else str(resseq)


def _build_gapped_alignment(seq_a, seq_b, aligned_coords):
    aligned_a = []
    aligned_b = []
    a_pos = 0
    b_pos = 0

    for (a_start, a_end), (b_start, b_end) in zip(*aligned_coords):
        if a_start > a_pos:
            aligned_a.append(seq_a[a_pos:a_start])
            aligned_b.append("-" * (a_start - a_pos))
        if b_start > b_pos:
            aligned_a.append("-" * (b_start - b_pos))
            aligned_b.append(seq_b[b_pos:b_start])

        aligned_a.append(seq_a[a_start:a_end])
        aligned_b.append(seq_b[b_start:b_end])
        a_pos = a_end
        b_pos = b_end

    if a_pos < len(seq_a):
        aligned_a.append(seq_a[a_pos:])
        aligned_b.append("-" * (len(seq_a) - a_pos))
    if b_pos < len(seq_b):
        aligned_a.append("-" * (len(seq_b) - b_pos))
        aligned_b.append(seq_b[b_pos:])

    return "".join(aligned_a), "".join(aligned_b)


def summarize_pdb_alignment(pdb_file, user_sequence, mutation_list=None, threshold_score=50):
    chain_data = _extract_pdb_chain_sequences(pdb_file)
    mutation_list = mutation_list or []
    mutation_positions = []
    for mut in mutation_list:
        match = re.search(r"(\d+)", mut)
        if match:
            mutation_positions.append((mut, int(match.group(1))))

    alignment_maps = {}

    for chain_id, chain_info in chain_data.items():
        pdb_seq = chain_info["seq"]
        residue_ids = chain_info["residue_ids"]
        hetero_resnames = chain_info["hetero_resnames"]
        if not pdb_seq:
            if hetero_resnames:
                print(
                    f"Chain {chain_id} has no protein residues. Non-protein residues: {', '.join(hetero_resnames)}"
                )
            else:
                print(f"Chain {chain_id} has no protein residues.")
            continue
        alignment = align_sequences(user_sequence, pdb_seq, mode="local", open_gap_score=-10, extend_gap_score=-0.5)
        if alignment.score < threshold_score:
            if hetero_resnames:
                print(
                    f"Chain {chain_id} skipped (alignment score {alignment.score:.1f}). Non-protein residues: {', '.join(hetero_resnames)}"
                )
            else:
                print(f"Chain {chain_id} skipped (alignment score {alignment.score:.1f}).")
            continue

        pdb_first_res = _format_res_id(residue_ids[0])
        pdb_last_res = _format_res_id(residue_ids[-1])
        print(f"PDB residue range (chain {chain_id}): {pdb_first_res}-{pdb_last_res}")

        user_indices, pdb_indices = _alignment_indices(alignment)
        if len(user_indices) == 0:
            if hetero_resnames:
                print(
                    f"Chain {chain_id} skipped (no aligned residues). Non-protein residues: {', '.join(hetero_resnames)}"
                )
            else:
                print(f"Chain {chain_id} skipped (no aligned residues).")
            continue

        user_min = min(user_indices) + 1
        user_max = max(user_indices) + 1
        pdb_min = min(pdb_indices)
        pdb_max = max(pdb_indices)
        pdb_min_res = residue_ids[pdb_min][1]
        pdb_max_res = residue_ids[pdb_max][1]

        print(f"Aligned region (chain {chain_id}): user {user_min}-{user_max} -> PDB {pdb_min_res}-{pdb_max_res}")

        user_to_pdb = dict(zip(user_indices, pdb_indices))

        if pdb_min > 0:
            pdb_prefix_seq = pdb_seq[:pdb_min]
            pdb_prefix_start = _format_res_id(residue_ids[0])
            pdb_prefix_end = _format_res_id(residue_ids[pdb_min - 1])
            print(
                f"Unaligned PDB prefix (chain {chain_id}, residues {pdb_prefix_start}-{pdb_prefix_end}): {pdb_prefix_seq}"
            )
        else:
            print(f"Unaligned PDB prefix (chain {chain_id}): none")

        if pdb_max + 1 < len(pdb_seq):
            pdb_suffix_seq = pdb_seq[pdb_max + 1 :]
            pdb_suffix_start = _format_res_id(residue_ids[pdb_max + 1])
            pdb_suffix_end = _format_res_id(residue_ids[-1])
            print(
                f"Unaligned PDB suffix (chain {chain_id}, residues {pdb_suffix_start}-{pdb_suffix_end}): {pdb_suffix_seq}"
            )
        else:
            print(f"Unaligned PDB suffix (chain {chain_id}): none")

        if user_min > 1:
            print(
                f"Unaligned user prefix (positions 1-{user_min - 1}): {user_sequence[:user_min - 1]}"
            )
        else:
            print("Unaligned user prefix: none")

        if user_max < len(user_sequence):
            print(
                f"Unaligned user suffix (positions {user_max + 1}-{len(user_sequence)}): {user_sequence[user_max:]}"
            )
        else:
            print("Unaligned user suffix: none")

        aligned_user, aligned_pdb = _build_gapped_alignment(
            user_sequence,
            pdb_seq,
            alignment.aligned,
        )

        alignment_maps[chain_id] = {
            "user_to_pdb": user_to_pdb,
            "residue_ids": residue_ids,
            "aligned_user": aligned_user,
            "aligned_pdb": aligned_pdb,
        }

    if mutation_positions:
        print("\nMutation mapping to PDB (by chain):")
        for mut, pos in mutation_positions:
            mapped = False
            for chain_id, chain_info in alignment_maps.items():
                user_to_pdb = chain_info["user_to_pdb"]
                residue_ids = chain_info["residue_ids"]
                user_idx = pos - 1
                if user_idx in user_to_pdb:
                    pdb_idx = user_to_pdb[user_idx]
                    pdb_resnum = residue_ids[pdb_idx][1]
                    print(f"  {mut}: chain {chain_id} -> PDB residue {pdb_resnum}")
                    mapped = True
            if not mapped:
                print(f"  {mut}: OUTSIDE aligned PDB region")

    return alignment_maps


def flag_outside_mutations(mutation_list, alignment_maps):
    mapped_positions = set()
    for chain_info in alignment_maps.values():
        if isinstance(chain_info, dict):
            mapped_positions.update(chain_info["user_to_pdb"].keys())
        else:
            mapped_positions.update(chain_info[0].keys())

    flagged = []
    for mut in mutation_list:
        match = re.search(r"(\d+)", mut)
        if not match:
            flagged.append(mut)
            continue
        pos = int(match.group(1)) - 1
        if pos in mapped_positions:
            flagged.append(mut)
        else:
            flagged.append(f"{mut} (OUTSIDE)")
    return flagged


if __name__ == "__main__":
    ids = list(sequences.keys())
    reference_id = ids[seq1_index]
    target_id = ids[seq2_index]
    print(f"Reference ID: {reference_id}")
    print(f"Target ID: {target_id}")
    user_seq = sequences[target_id]
    mutations = [
        m
        for m in get_mutations(sequences[reference_id], sequences[target_id])
        if "del" not in m and "-" not in m
    ]

    with open(reference_path, "r") as f:
        ref_record = next(SeqIO.parse(f, "fasta"))
    h3_map_with_ha2 = create_h3_numbering_map(user_seq, str(ref_record.seq), HA2_start=330)
    canonical_mutations = mutations_to_canonical(mutations, h3_map_with_ha2)
    print("Canonical mutations (first 10):", canonical_mutations[:10])

    alignment_maps = summarize_pdb_alignment(pdb_path, user_seq, mutation_list=mutations)
    mutations_flagged = flag_outside_mutations(mutations, alignment_maps)

    view = visualise_mutations_on_pdb(
        pdb_path,
        user_seq,
        mutations_flagged,
        title=f"{target_id} mutations",
        canonical_map=h3_map_with_ha2,
    )
    view.show()

    view_surface = visualise_mutations_on_pdb(
        pdb_path,
        user_seq,
        mutations_flagged,
        title=f"{target_id} mutations (surface)",
        canonical_map=h3_map_with_ha2,
        surface_opacity=0.7,
        surface_color="#dddddd",
        mutation_surface_opacity=1.0,
        mutation_surface_probe_radius=3.0,
        base_surface_exclude_mutations=True,
    )
    view_surface.show()

    view_surface_only = visualise_mutations_on_pdb(
        pdb_path,
        user_seq,
        mutations_flagged,
        title=f"{target_id} mutations (surface only)",
        canonical_map=h3_map_with_ha2,
        surface_opacity=1.0,
        surface_color="#dddddd",
        mutation_surface_opacity=1.0,
        mutation_surface_probe_radius=3.0,
        base_surface_exclude_mutations=True,
        hide_cartoon=True,
    )
    view_surface_only.show()

    # membrane_output_path = os.path.join(
    #     output_dir,
    #     f"{os.path.basename(pdb_path)}_membrane_embedded.pdb",
    # )
    # membrane_embed = embed_membrane_to_protein_plane(
    #     pdb_path,
    #     membrane_pdb_path,
    #     residue_start=526,
    #     residue_end=541,
    #     output_membrane_pdb_path=membrane_output_path,
    #     rotate_membrane=True,
    #     inplane_align=True,
    # )
    # membrane_resnames = membrane_embed["membrane_resnames"]

    # view_surface_membrane = visualise_mutations_on_pdb(
    #     pdb_path,
    #     user_seq,
    #     mutations_flagged,
    #     title=f"{target_id} mutations (surface + membrane)",
    #     canonical_map=h3_map_with_ha2,
    #     surface_opacity=0.7,
    #     surface_color="#dddddd",
    #     mutation_surface_opacity=1.0,
    #     mutation_surface_probe_radius=3.0,
    #     base_surface_exclude_mutations=True,
    # )
    # with open(membrane_output_path, "r") as f:
    #     view_surface_membrane.addModel(f.read(), "pdb")
    # if membrane_resnames:
    #     view_surface_membrane.addStyle(
    #         {"resn": membrane_resnames},
    #         {"stick": {"color": "#7fcdbb", "radius": 0.15}},
    #     )
    #     view_surface_membrane.addSurface(
    #         py3Dmol.SAS,
    #         {"opacity": 0.5, "color": "#7fcdbb"},
    #         {"resn": membrane_resnames},
    #     )
    # view_surface_membrane.show()

    pdb_name = os.path.basename(pdb_path)
    lineage_source = reference_id.split("|")[-1]
    lineage_target = target_id.split("|")[-1]

    surface_only_view_state = None
    surface_only_png_path = os.path.join(
        output_dir,
        f"{pdb_name}_{lineage_source}_to_{lineage_target}_mutations_surface_only.png",
    )
    export_view_to_png(
        view_surface_only,
        surface_only_png_path,
        width=3000,
        height=3000,
        view_state=surface_only_view_state,
        zoom_to=True,
    )

    chain_plddt = _extract_plddt_by_chain(pdb_path)
    plddt_sums = {}
    plddt_counts = {}
    for chain_id, chain_info in alignment_maps.items():
        user_to_pdb = chain_info["user_to_pdb"]
        plddt_list = chain_plddt.get(chain_id)
        if not plddt_list:
            continue
        for user_idx, pdb_idx in user_to_pdb.items():
            if pdb_idx >= len(plddt_list):
                continue
            plddt = plddt_list[pdb_idx]
            if plddt is None:
                continue
            pos_1based = user_idx + 1
            plddt_sums[pos_1based] = plddt_sums.get(pos_1based, 0.0) + plddt
            plddt_counts[pos_1based] = plddt_counts.get(pos_1based, 0) + 1

    plddt_background = {
        pos: plddt_sums[pos] / plddt_counts[pos]
        for pos in plddt_sums
        if plddt_counts.get(pos, 0) > 0
    }

    view_plddt = None
    if plddt_background:
        plddt_values = list(plddt_background.values())
        plddt_min = min(plddt_values)
        plddt_max = max(plddt_values)
        plddt_mean = sum(plddt_values) / len(plddt_values)
        plddt_title = (
            f"{target_id} pLDDT (mean {plddt_mean:.1f}, min {plddt_min:.1f}, max {plddt_max:.1f})"
        )
        view_plddt = visualise_mutations_on_pdb(
            pdb_path,
            user_seq,
            mutations_flagged,
            title=plddt_title,
            canonical_map=h3_map_with_ha2,
            background_values=plddt_background,
        )
        view_plddt.show()
    else:
        print("No pLDDT values detected in structure; skipping pLDDT ribbon plot.")

    
    output_path = os.path.join(
        output_dir,
        f"{pdb_name}_{lineage_source}_to_{lineage_target}_mutations_structure.html",
    )
    with open(output_path, "w") as f:
        f.write(view._make_html())
    print(f"Saved PDB plot to: {output_path}")

    surface_output_path = os.path.join(
        output_dir,
        f"{pdb_name}_{lineage_source}_to_{lineage_target}_mutations_surface.html",
    )
    with open(surface_output_path, "w") as f:
        f.write(view_surface._make_html())
    print(f"Saved surface plot to: {surface_output_path}")

    surface_only_output_path = os.path.join(
        output_dir,
        f"{pdb_name}_{lineage_source}_to_{lineage_target}_mutations_surface_only.html",
    )
    with open(surface_only_output_path, "w") as f:
        f.write(view_surface_only._make_html())
    print(f"Saved surface-only plot to: {surface_only_output_path}")

    # surface_membrane_output_path = os.path.join(
    #     output_dir,
    #     f"{pdb_name}_{lineage_source}_to_{lineage_target}_mutations_surface_membrane.html",
    # )
    # with open(surface_membrane_output_path, "w") as f:
    #     f.write(view_surface_membrane._make_html())
    # print(f"Saved surface+membrane plot to: {surface_membrane_output_path}")

    if view_plddt is not None:
        plddt_output_path = os.path.join(
            output_dir,
            f"{pdb_name}_{lineage_source}_to_{lineage_target}_plddt_ribbon.html",
        )
        with open(plddt_output_path, "w") as f:
            f.write(view_plddt._make_html())
        print(f"Saved pLDDT ribbon plot to: {plddt_output_path}")

    alignments_path = os.path.join(
        output_dir,
        f"{pdb_name}_{lineage_source}_to_{lineage_target}_alignments.fasta",
    )
    with open(alignments_path, "w") as f:
        for chain_id, chain_info in alignment_maps.items():
            aligned_user = chain_info["aligned_user"]
            aligned_pdb = chain_info["aligned_pdb"]
            f.write(f">user_{lineage_target}_chain_{chain_id}\n")
            f.write(f"{aligned_user}\n")
            f.write(f">pdb_{pdb_name}_chain_{chain_id}\n")
            f.write(f"{aligned_pdb}\n")
    print(f"Saved alignment FASTA to: {alignments_path}")
    
    
