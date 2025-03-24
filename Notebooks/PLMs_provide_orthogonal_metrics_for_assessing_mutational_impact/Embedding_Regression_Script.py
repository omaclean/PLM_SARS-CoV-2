
import torch
import esm
import pandas as pd
import numpy as np
from Functions import *

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import scipy
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR, LinearSVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_validate
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import RepeatedKFold
import matplotlib.pyplot as plt


ref_spike_seq = 'MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT'


dms_results=decompress_pickle('./DMS/SARS-CoV-2_3B_S/S/representations/DMS_S.pbz.pbz2')


def extract_representations(dms_results,region):
    reps = []
    for key in dms_results[region].keys():
        rep = pd.DataFrame([key,dms_results[region][key]['logits'], dms_results[region][key]['embeddings']],index=['label','Logits','Mean_Embedding'])
        reps.append(rep)
    return pd.concat(reps,axis=1).T


dms_representations = extract_representations(dms_results,'S')
dms_representations


dms_annotated = pd.read_csv('./DMS/DMS_S_annotated.csv')
dms_annotated


full_table = pd.merge(dms_annotated,dms_representations,how='left',left_on='label',right_on='label')
full_table = full_table[(full_table['label'].str.contains('mask') == False) & (full_table['ref']!= full_table['alt'])]
full_table

# %%
def group_by_amino_acid(predictor,feature,table):
    position_table = table[['pos',predictor]].drop_duplicates('pos')
    amino_acids = np.sort(table.ref.unique())
    feature_dict = {}
    for position in position_table.pos:
        feature_set = []
        for amino_acid in amino_acids:
            amino_acid_feature = table[(table.pos == position) & (table.alt == amino_acid)]
            if len(amino_acid_feature) == 0:
                amino_acid_feature = 0
            else:
                amino_acid_feature = amino_acid_feature[feature].values[0]
                if np.isnan(amino_acid_feature) == True:
                    amino_acid_feature = 0
            feature_set.append(amino_acid_feature)
        feature_dict[position] = feature_set
    df = pd.DataFrame(feature_dict).T
    df[feature] = df.values.tolist()
    df = df[[feature]]
    df = pd.merge(position_table,df,left_on='pos',right_index=True)
    return df

features = ['semantic_score',
            'relative_grammaticality',
            'relative_sequence_grammaticality',
            'relative_grammaticality_with_eve',
            'grammaticality',
            'masked_grammaticality',
            'mutated_grammaticality',
            
            'Evescape Fitness','Evescape',

            'IND Mutability Score','DCA Mutability Score',

             'Mean_Embedding','Logits'
             ]

predictors = ['Accessibility','PSSM','ESST','Entropy','Escape','RBD Escape','Entry','Binding','RBD Wuhan-Hu-1 Binding','RBD Variant Average Binding','RBD Wuhan-Hu-1 Expression','RBD Variant Average Expression','ΔΔG','B-Factor']

# %%
linear_best_predictor_results = []
new = True
for predictor in predictors:
    print('--------------------------------'+predictor.upper()+'--------------------------------')
    if predictor !='Entropy':
        data = full_table[full_table[predictor].isna() == False]
    else:
        data = full_table[full_table[predictor] >0]

    for feature in features:
        
        print('--------------------------------'+feature.upper()+'--------------------------------')
       
        if feature == 'semantic_score/relative_sequence_grammaticality':
            X = np.array([[data.semantic_score.values[i],data.relative_sequence_grammaticality.values[i]] for i in range(len(data.semantic_score.values))])
        elif feature == 'semantic_score/relative grammaticality':
            X = np.array([[data.semantic_score.values[i],data.relative_grammaticality.values[i]] for i in range(len(data.semantic_score.values))])
        elif feature == 'semantic_score':
            X = np.array([[data.semantic_score.values[i]] for i in range(len(data.semantic_score.values))])
        elif feature == 'relative grammaticality':
            X = np.array([[data.relative_grammaticality.values[i]] for i in range(len(data.semantic_score.values))])
        elif feature == 'relative_sequence_grammaticality':
            X = np.array([[data.relative_sequence_grammaticality.values[i]] for i in range(len(data.semantic_score.values))])
        if feature == 'IND Mutability Score' or feature == 'DCA Mutability Score' or feature == 'Evescape Fitness':
            X = data[data[feature].isna() == False]
            X =np.array(X[feature].to_list())
        else:
            X =np.array(data[feature].to_list())

        if predictor == 'Accessibility' or predictor == 'B-Factor' or predictor == 'Entropy' :
            if feature != 'Mean_Embedding' and feature != 'Logits':
                X = group_by_amino_acid(predictor,feature,data)
                X =np.array(list(X[feature]))

        y = data[predictor]
        
        train_size = 0.8
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)
        kf = RepeatedKFold(n_splits=5,n_repeats=3)
        for train_index , test_index in kf.split(X):
            X_train , X_test = X.iloc[train_index],X.iloc[test_index]
            y_train , y_test = y.iloc[train_index] , y.iloc[test_index]
            lin_preds = LinearRegression().fit(X_train, y_train).predict(X_test)
            lin_spearman = scipy.stats.spearmanr(y_test, lin_preds)
            linear_best_predictor_results.append(['Linear',lin_spearman.correlation,lin_spearman.pvalue])
            print(f'Linear:{lin_spearman}')
            
            scaler = StandardScaler()
            scaler = scaler.fit(X_train)
            scaled_Xtrain = scaler.transform(X_train)
            scaled_Xtest = scaler.transform(X_test)
            
            svr = SVR(kernel='rbf')
            svr.fit(scaled_Xtrain,np.ravel(y_train))
            svr_preds =svr.predict(scaled_Xtest)
            svr_spearman = scipy.stats.spearmanr(y_test, svr_preds)
            linear_best_predictor_results.append(['SVR-RBF',svr_spearman.correlation,svr_spearman.pvalue])
            
            print(f'SVR:{svr_spearman}')
            #Write results to file
            if new != True:
                pd.DataFrame(['Linear',lin_spearman.correlation,lin_spearman.pvalue,predictor,feature],index = ['model','correlation','pvalue','predictor','feature']).T.to_csv(
                    'DMS/Results/Regression/Model_Fitted_Correlations.csv', mode='a', index=False, header=False)
                pd.DataFrame(['SVR-RBF',svr_spearman.correlation,svr_spearman.pvalue,predictor,feature],index = ['model','correlation','pvalue','predictor','feature']).T.to_csv(
                    'DMS/Results/Regression/Model_Fitted_Correlations.csv', mode='a', index=False, header=False)
            else:
                pd.DataFrame(['Linear',lin_spearman.correlation,lin_spearman.pvalue,predictor,feature],index =['model','correlation','pvalue','predictor','feature']).T.to_csv(
                    'DMS/Results/Regression/Model_Fitted_Correlations.csv', index=False,)
                pd.DataFrame(['SVR-RBF',svr_spearman.correlation,svr_spearman.pvalue,predictor,feature],index = ['model','correlation','pvalue','predictor','feature']).T.to_csv(
                    'DMS/Results/Regression/Model_Fitted_Correlations.csv', mode='a', index=False, header=False)
                new=False

# %%
