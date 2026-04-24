import json
import re
import copy

def parse_gff3_for_cds(gff_path, gene_name="S"):
    """
    Parses the GFF3 file to find the start and end coordinates of a specific gene's CDS.
    """
    with open(gff_path, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            
            feature_type = parts[2]
            attributes = parts[8]
            
            # Look for the CDS of the requested gene
            if feature_type == "CDS" and f"gene={gene_name}" in attributes:
                # GFF coordinates are 1-indexed and inclusive
                start = int(parts[3])
                end = int(parts[4])
                return start, end
    raise ValueError(f"Could not find CDS for gene {gene_name} in GFF3 file.")

def parse_nextclade_tree_nuc(json_path):
    """
    Traverses the Auspice tree JSON to accumulate nucleotide mutations
    from the root to each terminal node.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    tree = data.get('tree', {})
    terminal_nodes = {}

    def traverse(node, current_muts):
        node_muts = copy.deepcopy(current_muts)
        
        # Extract nucleotide mutations (e.g., C241T, T11288-)
        branch_muts = node.get("branch_attrs", {}).get("mutations", {}).get("nuc", [])
        for mut in branch_muts:
            match = re.match(r"([A-Z\-\*])(\d+)([A-Z\-\*])", mut)
            if match:
                ref, pos, alt = match.groups()
                node_muts[int(pos)] = alt

        node_name = node.get("name", "Unknown")

        if "children" not in node or not node["children"]:
            terminal_nodes[node_name] = node_muts
        else:
            for child in node.get("children", []):
                traverse(child, node_muts)

    traverse(tree, {})
    return terminal_nodes

def generate_spike_cds_alignments(lineage_data, ref_seq, gff_coords, out_full, out_no_gaps):
    """
    Applies mutations to the reference genome, extracts the Spike CDS based on 
    GFF coordinates, and writes out the two FASTA formats.
    """
    start, end = gff_coords
    # Convert to 0-indexed for Python slicing
    start_idx = start - 1
    end_idx = end 
    
    # Extract the reference Spike directly to use as the base template
    ref_spike = ref_seq[start_idx:end_idx]

    with open(out_full, 'w') as f_full, open(out_no_gaps, 'w') as f_no_gaps:
        # Write the Wuhan-Hu-1 Reference to both files
        # f_full.write(f">Wuhan-Hu-1_Reference\n")
        # for i in range(0, len(ref_spike), 80):
        #     f_full.write(f"{ref_spike[i:i+80]}\n")
        f_full.write(f">Wuhan-Hu-1_Reference\n{ref_spike}\n")
        # f_no_gaps.write(f">Wuhan-Hu-1_Reference\n")
        # for i in range(0, len(ref_spike), 80):
        #     f_no_gaps.write(f"{ref_spike[i:i+80]}\n")
        f_no_gaps.write(f">Wuhan-Hu-1_Reference\n{ref_spike}\n")
        for name, muts in lineage_data.items():
            # Convert ref genome to list to apply mutations globally
            seq_list = list(ref_seq)
            
            for pos, alt in muts.items():
                if 1 <= pos <= len(seq_list):
                    seq_list[pos - 1] = alt
            
            mutated_full_genome = "".join(seq_list)
            
            # Extract just the Spike region
            mutated_spike = mutated_full_genome[start_idx:end_idx]
            
            # Remove gap characters (-) to create the continuous sequence
            gapless_spike = mutated_spike.replace("-", "")
            
            # f_full.write(f">{name}\n")
            # for i in range(0, len(mutated_spike), 80):
            #     f_full.write(f"{mutated_spike[i:i+80]}\n")
                
            # f_no_gaps.write(f">{name}\n")
            # for i in range(0, len(gapless_spike), 80):
            #     f_no_gaps.write(f"{gapless_spike[i:i+80]}\n")
            f_full.write(f">{name}\n{mutated_spike}\n")
            
            f_no_gaps.write(f">{name}\n{gapless_spike}\n")
if __name__ == "__main__":
    input_json = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/SC2_tree.json"
    gff_file = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/wuhan-hu-1-sequence.gff3"
    
    output_prefix = '/home3/oml4h/PLM_SARS-CoV-2/Sequences/SC2_lineage_references'
    out_aligned_spike = f"{output_prefix}_spike_nt_aligned.fasta"
    out_gapless_spike = f"{output_prefix}_spike_nt_no_gaps.fasta"
    
    # Paste the full 29903 bp Wuhan-Hu-1 reference genome string here.
    with open("/home3/oml4h/PLM_SARS-CoV-2/Sequences/Wuhan_full.fa", "r") as f:
        wuhan_genome_ref = f.read().strip().split("\n", 1)[1].replace("\n", "")
    
    print("Parsing GFF3 for Spike coordinates...")
    spike_coords = parse_gff3_for_cds(gff_file, gene_name="S")
    print(f"Spike CDS found at coordinates: {spike_coords[0]} - {spike_coords[1]}")
    
    print("Analysing tree and accumulating nucleotide mutations...")
    extracted_data = parse_nextclade_tree_nuc(input_json)
    print(f"Extracted data for {len(extracted_data)} terminal lineages.")
    
    print("Generating Spike CDS FASTA files...")
    generate_spike_cds_alignments(
        extracted_data, 
        wuhan_genome_ref, 
        spike_coords, 
        out_aligned_spike, 
        out_gapless_spike
    )
    
    print(f"Exported aligned Spike CDS to {out_aligned_spike}")
    print(f"Exported gapless Spike CDS to {out_gapless_spike}")