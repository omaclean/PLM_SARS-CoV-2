######################################### Pytorch imports ################################################## 
import torch
from torch.utils.data import DataLoader
######################################### Transformers imports ############################################## 
# Use a pipeline as a high-level helper
from transformers import pipeline
# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM
######################################### Numpy and Pandas imports ############################################
import numpy as np
import pandas as pd
######################################### Biopython imports ##################################################
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import Entrez
from Bio import SeqIO
######################################### SciPy imports ##################################################
#from scipy.special import softmax
######################################### tqdm imports ##################################################
from tqdm import tqdm

## Compression Functions ########################################################
import bz2
import pickle
import _pickle as cPickle
def compressed_pickle(title, data):
  with bz2.BZ2File(title + '.pbz2', 'w') as f:
    cPickle.dump(data, f)

def decompress_pickle(file):
  data = bz2.BZ2File(file, 'rb')
  data = cPickle.load(data)
  return data

def mutate_sequence(reference_sequence,mutations):
    mutated_seq = reference_sequence
    for mutation in mutations:
        if 'ins' not in mutation and 'del' not in mutation and "X" not in mutation:
            mutant_amino = mutation[-1]
            mutant_pos = int(mutation[1:-1])
            mutated_seq = mutated_seq[:mutant_pos-1]+mutant_amino+mutated_seq[mutant_pos:]
    return mutated_seq

def DMS(reference,start=0,end = None,mask=False):
  if end == None:
    end = len(reference)
  seq_list = []
  if mask == False:
    amino_acids = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
  else:
    amino_acids = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V","<mask>"]
  for i,ref_amino_acid in enumerate(reference):
      if i>=start and i<=end:
        for mutant_amino_acid in amino_acids:
            mutated_seq = reference[:i]+mutant_amino_acid+reference[i+1:]
            seq = SeqRecord(Seq(mutated_seq), id=ref_amino_acid+str(i+1)+mutant_amino_acid)
            seq_list.append(seq)
  return seq_list

def Batch_DMS(reference,start=0,end = None,mask=False):
    #Convert DMS sequences into ids and sequences rather than seq records
    ids,seqs = zip(*[[s.id,str(s.seq)] for s in DMS(reference,mask=mask)])
    #Extend batch size if mask is used
    if mask == True:
        batch_size=21
    else:
        batch_size=20
    #Batch sequences into appropriate sized batches 
    seqs_batched = [ seqs[i:i + batch_size] for i in range(0, len(seqs), batch_size)]
    ids_batched = [ ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]
    return ids_batched ,seqs_batched

def embed_batch(sequences,tokenizer,model,device):
    #Initialise logsoftmax function
    batch_lsoftmax = torch.nn.LogSoftmax(dim=2)
    
    # Get logits for the masked token
    with torch.no_grad():
        
        # Tokenize the input sequences
        tokenized_batch_sequences = tokenizer(sequences,return_tensors="pt").to(device)
        sequence_length = len(sequences[0])
        
        #Embed Sequences
        output = model(**tokenized_batch_sequences)
        
        #Extract logits and remove <cls> and <eos> tokens.
        logits = output.logits[:,1:sequence_length+1,:].to('cpu')
        
        #Softmax logits
        logits = np.array(batch_lsoftmax(logits))
        
        #Extract embeddings, take the final layer, remove <cls> and <eos> and mean.
        embeddings = np.array([e.mean(axis=0) for e in output.hidden_states[-1][:,1:sequence_length+1,:].to('cpu')])
        
        tokenized_batch_sequences = tokenized_batch_sequences.to('cpu')['input_ids'][:,1:-1]
        
        torch.cuda.empty_cache()
        
    return np.array(logits),np.array(embeddings),np.array(tokenized_batch_sequences)

def makeOrfTable(genbank_record):
    orfs=[]
    for feature in genbank_record.features:
        print(feature)
        if feature.type =="CDS":
            print(feature.qualifiers.keys())
            orf = feature.qualifiers['gene'][0]
            for i, locations in enumerate(feature.location.parts):
                orfs.append([orf, locations.start, locations.end, i, locations])
    orfs = pd.DataFrame(orfs)
    orfs.columns = ['ORF','Start','End','Part','Locations']
    orfs = orfs.set_index("ORF")
    return orfs

def makeMatProteinTable(genbank_record):
    proteins=[]
    for feature in genbank_record.features:
        if feature.type =="mat_peptide":
            protein = feature.qualifiers['product'][0]
            orf = feature.qualifiers['gene'][0]
            for i, locations in enumerate(feature.location.parts):
                proteins.append([protein, orf ,locations.start, locations.end, i, locations])
    proteins = pd.DataFrame(proteins)
    if len(proteins) == 0:
        return pd.DataFrame(columns=['Protein',"ORF",'Start','End','Part','Locations'])
    proteins.columns = ['Protein',"ORF",'Start','End','Part','Locations']
    proteins = proteins.set_index("Protein")
    return proteins

def iterative_translate(sequence,truncate_proteins=False):
    amino_acid = ""
    for i in range(0,len(sequence)-2,3):
        codon = str(sequence[i:i+3])
        codon = codon.replace("?", "N")
        if "-" in codon:
            if codon == "---":
                amino_acid +="-"
            else:
                amino_acid+= "X"
        else:
            amino_acid += str(Seq(codon).translate())
    if truncate_proteins == True:
        if "*" in amino_acid:
            amino_acid = amino_acid[:amino_acid.index("*")]
    return amino_acid

def translate_with_genbank(sequence,ref):
    orfs = makeOrfTable(ref)
    print(orfs)
    translated_sequence = {orfs.index[i]+":"+str(orfs.iloc[i].Part):{"Sequence":"".join(list(iterative_translate("".join(orfs.iloc[i].Locations.extract(sequence)),truncate_proteins=True))),"ORF":orfs.index[i]} for i in range(len(orfs))}
    return translated_sequence

def translate_mat_proteins_with_genbank(sequence,ref):
    proteins = makeMatProteinTable(ref)
    proteins = proteins.drop_duplicates(subset=["ORF",'Start','End','Part',],keep="first")
    proteins_dict={}
    for i in range(len(proteins)):
        protein = "".join(list(iterative_translate("".join(proteins.iloc[i].Locations.extract(sequence)),truncate_proteins=True)))
        if proteins.index[i] in proteins_dict:
            proteins_dict[proteins.index[i]]["Sequence"] = proteins_dict[proteins.index[i]]["Sequence"]+protein
        else:
            proteins_dict[proteins.index[i]] = {"Sequence":protein, "ORF":proteins.iloc[i].ORF, "Part":proteins.iloc[i].Part}
    # translated_sequence = {proteins.index[i]:{"Sequence":"".join(list(iterative_translate("".join(proteins.iloc[i].Locations.extract(sequence)),truncate_proteins=True))), "ORF":proteins.iloc[i].ORF} }
    return proteins_dict

def process_sequence_genbank(sequence,genbank):
    #Translate nucleotide to proteins using genbank
    Coding_Regions= translate_with_genbank(sequence,genbank)
    Mature_Proteins= translate_mat_proteins_with_genbank(sequence,genbank)
    polyprotein_orfs =set([Mature_Proteins[prot]["ORF"] for prot in Mature_Proteins.keys()])
    Filtered_Coding_Regions = {**Coding_Regions}
    for orf in Coding_Regions.keys():
        if Coding_Regions[orf]["ORF"] in polyprotein_orfs:
            del Filtered_Coding_Regions[orf]
    Merged_Coding_Regions = {**Filtered_Coding_Regions,**Mature_Proteins}
    return Merged_Coding_Regions

def rescale_reverse_bound(series):
    """
    Rescales a series to make all numbers positive, reverses the order,
    and bounds the values between 0 and 1.

    Parameters:
        series (list or array): A list of numbers (positive and/or negative).

    Returns:
        list: A transformed list where all numbers are in [0, 1], with the
              largest reversed to the smallest.
    """
    # Step 1: Rescale to make all numbers positive
    min_value = min(series)
    shift = abs(min_value)
    rescaled = [x + shift for x in series]
    
    # Step 2: Normalize to [0, 1]
    max_value = max(rescaled)
    normalized = [x / max_value for x in rescaled]
    
    # Step 3: Reverse within [0, 1]
    reversed_scaled = [1 - x for x in normalized]
    
    return reversed_scaled
