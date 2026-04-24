import json
import re
import csv
import copy

def parse_nextclade_tree(json_path):
    """
    Traverses the Auspice tree JSON to accumulate Spike mutations
    from the root to each terminal node (variant/lineage).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # The root node of the phylogenetic tree
    tree = data.get('tree', {})
    terminal_nodes = {}

    def traverse(node, current_muts):
        # Deep copy ensures branches do not share downstream mutation states
        node_muts = copy.deepcopy(current_muts)
        
        # Extract S mutations occurring precisely on this branch
        branch_muts = node.get("branch_attrs", {}).get("mutations", {}).get("S", [])
        for mut in branch_muts:
            # Regex captures Reference AA, Position, and Alternate AA (e.g., D614G or H69-)
            match = re.match(r"([A-Z\-\*])(\d+)([A-Z\-\*])", mut)
            if match:
                ref, pos, alt = match.groups()
                node_muts[int(pos)] = alt

        node_name = node.get("name", "Unknown")

        # If it's a terminal leaf node (no children), save the accumulated profile
        if "children" not in node or not node["children"]:
                terminal_nodes[node_name] = node_muts
        else:
            # Continue traversing down the branches
            for child in node.get("children", []):
                traverse(child, node_muts)

    traverse(tree, {})
    return terminal_nodes

def export_mutations_to_csv(lineage_data, output_csv="lineage_spike_mutations.csv"):
    """
    Writes the accumulated mutations for each lineage to a CSV.
    """
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Lineage_Name", "Total_S_Mutations", "Mutation_List"])
        
        for name, muts in lineage_data.items():
            # Reconstruct standard mutation nomenclature (e.g., 614G)
            mut_list = [f"{pos}{alt}" for pos, alt in sorted(muts.items())]
            writer.writerow([name, len(mut_list), ", ".join(mut_list)])

def generate_fasta_alignment(lineage_data, ref_seq, output_fasta="spike_alignment.fasta"):
    """
    Applies the mutation dictionary to the reference sequence to build
    an aligned FASTA file.
    """
    with open(output_fasta, 'w') as f:
        # Write the Wuhan reference first
        f.write(">Wuhan-Hu-1_Reference\n")
        f.write(f"{ref_seq}\n")
        
        for name, muts in lineage_data.items():
            # Convert string to list for mutable indexing
            seq_list = list(ref_seq)
            
            for pos, alt in muts.items():
                # Mutations are 1-indexed; Python lists are 0-indexed
                if 1 <= pos <= len(seq_list):
                    seq_list[pos - 1] = alt
            
            mutated_seq = "".join(seq_list)
            f.write(f">{name}\n")
            f.write(f"{mutated_seq}\n")

if __name__ == "__main__":
    # 1. Define your input JSON file path
    input_json = "/home3/oml4h/PLM_SARS-CoV-2/Sequences/SC2_tree.json"
    output_prefix ='/home3/oml4h/PLM_SARS-CoV-2/Sequences/SC2_lineage_references'
    output_fasta=f"{output_prefix}_spike_alignment.fasta"
    output_csv=f"{output_prefix}_lineage_spike_mutations.csv"
    
    
    # 2. Paste the exact 1273 amino acid Wuhan-Hu-1 S protein sequence here
    # (Truncated below for demonstration; replace with the full sequence)
    wuhan_s_ref = (
     "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
     )
    
    # Run the extraction and generation
    print("Analysing tree and accumulating mutations...")
    extracted_data = parse_nextclade_tree(input_json)
    
    print(f"Extracted data for {len(extracted_data)} terminal lineages.")
    
    export_mutations_to_csv(extracted_data, output_csv)
    print("Exported mutations to lineage_spike_mutations.csv")
    
    generate_fasta_alignment(extracted_data, wuhan_s_ref, output_fasta)
    print("Exported alignment to spike_alignment.fasta")