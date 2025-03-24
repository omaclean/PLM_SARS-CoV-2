############################################################################################################
######################################### Imports ##########################################################
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
############################################################################################################
####################################### Functions ##########################################################
from HuggingFace_Functions import *
import os
############################################################################################################
################################ Input parameters ##########################################################
params = {
    #Specify ORF to analyse
    'specify_orf':"S",
    #Decide if <mask> is used in DMS
    'include_mask_token_in_DMS' : True,
    #Select model
    'model_name' : "facebook/esm2_t36_3B_UR50D",
    #Specify if ids are from genbank or are protein names
    'use_genbank':False,
    #Specify top directory to save DMS
    'container_directory':"SARS-CoV-2",
    #Specify if genbank is a virus assembly (currently just influenza viruses)
    'ncbi_virus_assembly':False,
    'save_representations':True
}


############################################################################################################
################################ Global Variables ##########################################################
#Amino acid tokens
amino_acids = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
############################################################################################################
################################ Load Model into GPU #######################################################
#Load model and tokenizer
model = AutoModelForMaskedLM.from_pretrained(params['model_name'],output_hidden_states = True)
tokenizer = AutoTokenizer.from_pretrained(params['model_name'])

#Assign device for using model (GPU)
device = torch.device("cuda:0")
model = model.to(device)
################################################################################################################
############################################## Main ############################################################

id_list = {  
    'S':"MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT",
}

if params['ncbi_virus_assembly'] == True:
    assembly = []

for identifier,query in id_list.items():
    #Create directory for protein/genbank file
    os.makedirs(params['container_directory']+'/'+identifier, exist_ok=True)
    if params['save_representations'] == True:
        os.makedirs(params['container_directory']+'/'+identifier+'/representations', exist_ok=True)
    #Use genbank file for DMS or just single sequence
    if params['use_genbank'] == True:
        #Retrieve genbank file
        Entrez.email = "sample@example.org"
        handle = Entrez.efetch(db="nucleotide",id=query,rettype="gb",retmode="gb")
        genbank_file = SeqIO.read(handle, "genbank")

        #Translate nucleotide to proteins using genbank
        Coding_Regions= translate_with_genbank(genbank_file.seq,genbank_file)
        print(Coding_Regions)
        Mature_Proteins= translate_mat_proteins_with_genbank(genbank_file.seq,genbank_file)
        if len(Mature_Proteins) != 0:
            polyprotein_orfs =set([Coding_Regions[prot]["ORF"] for prot in Coding_Regions.keys()])
            polyprotein_orfs =set([Mature_Proteins[prot]["ORF"] for prot in Mature_Proteins.keys()])
            Filtered_Coding_Regions = {**Coding_Regions}
            for orf in Coding_Regions.keys():
                if Coding_Regions[orf]["ORF"] in polyprotein_orfs:
                    del Filtered_Coding_Regions[orf]
            Merged_Coding_Regions = {**Filtered_Coding_Regions,**Mature_Proteins}
        else:
            Merged_Coding_Regions = Coding_Regions
        embeddings = {}
        if params['specify_orf'] !="":
            Merged_Coding_Regions = {params['specify_orf']:Merged_Coding_Regions[params['specify_orf']]}
    else:
        Merged_Coding_Regions = {identifier:{"Sequence":query}}

    ############################################################################################################
    ############################################## Inner Main ########################################################
    all_dfs = []

    

    
    #Embed Sequence
    for key,value in Merged_Coding_Regions.items():

        #Create dictionary for logits and embeddings
        representations = {}
        representations[key] = {}
        print(value['Sequence'])
        
        dfs = []
        # Define the reference sequence
        reference_sequence = value['Sequence']

        #Define sequence length
        sequence_length = len(reference_sequence)

        #Perform and batch DMS on reference sequence
        ids,sequences = Batch_DMS(reference_sequence,mask=params['include_mask_token_in_DMS'])

        #Calculate reference logits and embedding
        reference_logits,reference_embedding,reference_tokens = embed_batch([reference_sequence],tokenizer,model,device)
        reference_logits,reference_embedding,reference_tokens =reference_logits[0],reference_embedding[0],reference_tokens[0]
        reference_sequence_grammaticality = np.sum(reference_logits[np.arange(reference_logits.shape[0]), reference_tokens])


        #Embed sequences in batches (one batch is equivelent to all the amino acids at a position in the sequence)
        for batch_number in tqdm(range(0,len(sequences))):
            #Index batch
            sequence_zero_position = batch_number

            batch_ids = ids[batch_number]
            batch_sequences = sequences[batch_number]

            #Embed batch and extract logits and mean embeddings
            logits,embeddings,tokens = embed_batch(batch_sequences,tokenizer,model,device)

            #Calculate semantic scores
            semantic_score = abs(embeddings - reference_embedding).sum(axis=1)
            
            logit_index =logits[:,sequence_zero_position,:]

            #Retrieve reference grammaticality (Log-likelihood of reference amino acid)
            reference_grammaticality = reference_logits[sequence_zero_position,reference_tokens[sequence_zero_position]]

            #Retrieve grammaticality (Log-likelihood of mutant amino acid)
            grammaticality = reference_logits[sequence_zero_position,tokens[:,sequence_zero_position]]
           
            #Calculate relative grammaticality (LLR)
            relative_grammaticality = grammaticality - reference_grammaticality

            sequence_logits = np.array([logits[:,i,:][np.arange(logit_index.shape[0]), tokens[:,i]]  for i in range(0,sequence_length)])

            #Calculate sequence grammaticality
            sequence_grammaticality = np.sum(sequence_logits,axis=0)

            #Calculate relative sequence grammaticality (PLLR)
            relative_sequence_grammaticality = sequence_grammaticality - reference_sequence_grammaticality

            #Calculate mutated_grammaticality
            mutated_grammaticality = logit_index[np.arange(logit_index.shape[0]), tokens[:,sequence_zero_position]]

            #Calculate relative_mutated_grammaticality (Reference sequence logits for sequence minus mutated sequence logits for sequence)
            relative_mutated_grammaticality = mutated_grammaticality - reference_grammaticality
            
            #<mask> grammaticalities
            if params['include_mask_token_in_DMS'] == True:
                #<mask> token is the last sequence in the DMS batch
                masked_grammaticality = logit_index[logit_index.shape[0]-1,tokens[:,sequence_zero_position]]
                relative_masked_grammaticality = masked_grammaticality - reference_grammaticality
            else:
                masked_grammaticality = np.arange(20,np.nan)
                relative_masked_grammaticality = np.arange(20,np.nan)

            print(sequence_logits.shape)
            #Save Representations (embeddings and Logits)
            if params['save_representations'] == True:
                for i in range(0,len(batch_ids)):
                    representations[key][batch_ids[i]] = {}
                    representations[key][batch_ids[i]]["logits"] = sequence_logits[:,i]
                    representations[key][batch_ids[i]]["embeddings"] = embeddings[i]
    
            #Append to dataframe
            df = pd.DataFrame([batch_ids, semantic_score, relative_grammaticality, relative_sequence_grammaticality,relative_mutated_grammaticality,relative_masked_grammaticality,grammaticality,sequence_grammaticality,mutated_grammaticality,masked_grammaticality,np.exp(grammaticality),np.exp(mutated_grammaticality),np.exp(masked_grammaticality)],
                            index = ['label','semantic_score','relative_grammaticality','relative_sequence_grammaticality','relative_mutated_grammaticality','relative_masked_grammaticality','grammaticality','sequence_grammaticality','mutated_grammaticality','masked_grammaticality','probability','mutated_probability','masked_probability']).T
            df['ref'] = df.label.str[0]
            df['alt'] = ['<mask>' if '<mask>' in i else i[-1] for i in  df.label]
            df['position'] = [int(i[1:-len('<mask>')]) if '<mask>' in i else int(i[1:-1]) for i in  df.label]
            df['region'] = key
            df['subunit'] = ""
            df['domain']  = ""
            df['GenbankID_or_Pro']  = identifier
            df['reference_grammaticality'] = reference_grammaticality
            dfs.append(df)
        dms_table = pd.concat(dfs,axis=0)
        dms_table = dms_table.sort_values('semantic_score')
        dms_table['semantic_rank'] = dms_table.reset_index().index.astype(int) + 1
        dms_table = dms_table.sort_values('grammaticality')
        dms_table['grammatical_rank'] =dms_table .reset_index().index.astype(int) + 1
        dms_table['acquisition_priority'] = dms_table['semantic_rank'] + dms_table['grammatical_rank']

        dms_table = dms_table.sort_values('sequence_grammaticality')
        dms_table['sequence_grammatical_rank'] =dms_table.reset_index().index.astype(int) + 1
        dms_table['sequence_acquisition_priority'] = dms_table['semantic_rank'] + dms_table['sequence_grammatical_rank']
        dms_table.to_csv(params['container_directory']+'/'+identifier+'/'+'DMS_'+key+'.csv',index=False)
        all_dfs.append(dms_table)
        compressed_pickle(params['container_directory']+'/'+identifier+'/representations/DMS_'+key+'.pbz',representations)


    ############################################################################################################
    ################################ Output DMS as DataFrame ###################################################
    all_dfs =  pd.concat(all_dfs ,axis=0)
    all_dfs.to_csv(params['container_directory']+'/'+identifier+"_DMS_scores.csv",index=False)
    if params['ncbi_virus_assembly'] == True:
        assembly.append(all_dfs)
if params['ncbi_virus_assembly'] == True:
    assembly = pd.concat(assembly,axis=0)
    assembly.to_csv(params['container_directory']+'/'+params['container_directory']+"_Assembly_DMS_scores.csv",index=False)
