#***********************************
# Run the command 'python data_preprocess.py' in the Terminal to start the data pre-procession.
# Or directly run this file in the Pycharm IDE.
# This may take several minutes.
# When is done, the files of prepared data are saved in the subfolders of cancer_folder.
# There should be several subfolders under this folder, named as follows,
# 'mut_similarity' saving the file of Gaussian interaction profile kernel similarity between mutated genes,
# 'orig_data' saving the RNA-seq.txt and SomaticMutation.txt file downloaded from TCGA by Xena,
# 'results' saving the file predicted by IMCDriver consisting of scores of mutated genes of each patient in 'Example'
# 'sample_similarity' saving the file of Gaussian interaction profile kernel similarity between samples,
#***********************************

# The folder name of your cancer dataset, such as 'Example' or 'BRCA'.
# You can change it to fit your data.
cancer_folder='Example'

import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler

def get_row_intersection_of_dataframe(df1,df2,df3):
    r=pd.DataFrame(df1.index.values.tolist(),columns=['rna'])
    s=pd.DataFrame(df2.index.values.tolist(),columns=['snp'])
    if df3 is not None:
        p=pd.DataFrame(df3.iloc[:,0].values.tolist(),columns=['ppi'])
        rp=pd.merge(left=r,right=p,left_on='rna',right_on='ppi',how='inner')
        rps=pd.merge(left=rp,right=s,left_on='rna',right_on='snp',how='inner')
    else:
        rps = pd.merge(left=r, right=s, left_on='rna', right_on='snp', how='inner')
    g_lst=rps['rna'].values.tolist()
    return g_lst

def get_col_intersection_of_dataframe(df1,df2):
    pls1=df1.columns.values.tolist()
    pls2=df2.columns.values.tolist()

    p_lst=[]
    for p in pls1:
        if p in pls2:
           p_lst.append(p)
    return p_lst

def filter_ppi_with_intersect_nodes(nodes_lst,ppi_df):
    g_lst_df=pd.DataFrame(nodes_lst,columns=['g1'])
    m1=pd.merge(left=ppi_df,right=g_lst_df,left_on='source',right_on='g1',how='left')
    m1.dropna(how='any',inplace=True)
    m1.drop(['g1'],axis=1,inplace=True)

    m2=pd.merge(left=m1,right=g_lst_df,left_on='target',right_on='g1',how='left')
    m2.dropna(how='any',inplace=True)
    m2.drop(['g1'],axis=1,inplace=True)
    return m2

def cal_outlying_gene_lst(df):
    expr_df_T = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
    test=StandardScaler().fit_transform(expr_df_T)
    expr_df = pd.DataFrame(test.T, index=df.index, columns=df.columns)
    otly_g=[]
    expr_df_abs=expr_df.abs()
    for g,row in expr_df_abs.iterrows():
        if (row>2).any():
            otly_g.append(g)
    return otly_g

def prepare_intersection_data(cancerType):
    ppi,ppi_nodes=load_network('./data/Additional_file5_reliable_interactions.txt')
    rna_file='./data/%s/orig_data/RNA-seq.txt' % cancerType
    snp_file='./data/%s/orig_data/SomaticMutation.txt' % cancerType
    rna_df=pd.read_table(filepath_or_buffer=rna_file,header=0,index_col=0,sep='\t')
    snp_df=pd.read_table(filepath_or_buffer=snp_file,header=0,index_col=0,sep='\t')

    g_lst=get_row_intersection_of_dataframe(df1=rna_df,df2=snp_df,df3=ppi_nodes)

    r_g_inter=rna_df.loc[g_lst,:]
    s_g_inter=snp_df.loc[g_lst,:]

    p_lst=get_col_intersection_of_dataframe(r_g_inter,s_g_inter)

    rna_inter_df=r_g_inter.loc[:,p_lst]
    snp_inter_df=s_g_inter.loc[:,p_lst]

    rna_inter_df.to_csv(path_or_buf='./data/%s/RNA-seq.txt' % cancerType , sep='\t', header=True,index=True)
    snp_inter_df.to_csv(path_or_buf='./data/%s/SNP.txt' % cancerType , sep='\t', header=True,index=True)

    ppi=filter_ppi_with_intersect_nodes(g_lst,ppi)
    ppi.to_csv(path_or_buf='./data/%s/PPI.txt' % cancerType , sep='\t', header=False,index=False)

def create_mutation_and_driver_matrices(cancerType):
    rna_df = pd.read_table(filepath_or_buffer='./data/%s/RNA-seq.txt' % cancerType, header=0, index_col=0, sep='\t')
    snp_df = pd.read_table(filepath_or_buffer='./data/%s/SNP.txt' % cancerType, header=0, index_col=0, sep='\t')

    samp_lst = rna_df.columns.values.tolist()
    dic_p = {}

    for id, row in snp_df.iterrows():
        if row.name not in dic_p.keys():
            dic_p[row.name] = list(np.abs(row.values))
        else:
            print('error in cnv_df, duplicate mutation.')

    mut_t1 = pd.DataFrame(dic_p, index=samp_lst)
    mut_P = pd.DataFrame(mut_t1.values.T, index=mut_t1.columns, columns=mut_t1.index)
    mut_P.to_csv(path_or_buf='./data/%s/mutation_P.txt' % cancerType, sep='\t', header=True, index=True)

    gold_drivers = pd.read_table(filepath_or_buffer='./data/NCG_known_711.txt', header=None, index_col=None,names=['name'])
    for m in dic_p.keys():
        if m not in list(gold_drivers['name'].values):
            dic_p[m] = [0 for i in np.arange(0, len(samp_lst))]

    t1 = pd.DataFrame(dic_p, index=samp_lst)
    P_orig = pd.DataFrame(t1.values.T, index=t1.columns, columns=t1.index)

    otly_g_ls = cal_outlying_gene_lst(rna_df)
    ppi, ppi_nodes = load_network('./data/%s/PPI.txt' % cancerType)
    remv_mut_ls = []
    for g in otly_g_ls:
        if g not in ppi_nodes['nodes'].values.tolist():
            remv_mut_ls.append(g)
    t1 = np.sum(mut_P.values, axis=1)
    P = P_orig.iloc[t1.nonzero()[0], :]
    for g in remv_mut_ls.copy():
        if g not in P.index.values.tolist():
            remv_mut_ls.remove(g)

    P = P.drop(remv_mut_ls, axis=0)
    P.to_csv(path_or_buf='./data/%s/P_filtered.txt' % cancerType, sep='\t', header=True, index=True)

# Calculate the Gaussian interaction profile kernel similarity between samples.
def calculate_samp_gaussian_similarity(arr_P):
    arr_P = np.array(arr_P)
    ns = arr_P.shape[1]

    ssm = np.zeros((ns, ns))
    sm = np.zeros((1, ns))

    for i in np.arange(0, ns):
        sm[0, i] = math.pow(np.linalg.norm(arr_P[:, i]), 2)
    gama = ns / np.sum(sm)

    for i in np.arange(0, ns):
        for j in np.arange(i, ns):
            ssm[i, j] = math.exp(-gama * math.pow(np.linalg.norm(arr_P[:, i] - arr_P[:, j]), 2))
        print('%d/%d' % (i+1,ns))

    ssm=ssm+ssm.transpose()

    for i in np.arange(0,ssm.shape[0]):
        ssm[i,i]=ssm[i,i]/2

    return ssm
# Calculate the Gaussian interaction profile kernel similarity between mutated genes.
def calculate_mut_gaussian_similarity(arr_P):
    arr_P = np.array(arr_P)
    ng = arr_P.shape[0]

    gsm = np.zeros((ng, ng))
    sm = np.zeros((1, ng))

    for i in np.arange(0, ng):
        sm[0, i] = math.pow(np.linalg.norm(arr_P[i, :]), 2)
    gama = ng / np.sum(sm)

    for i in np.arange(0, ng):
        for j in np.arange(i, ng):
            gsm[i, j] = math.exp(-gama * math.pow(np.linalg.norm(arr_P[i, :] - arr_P[j, :]), 2))
        print('%d/%d' % (i+1,ng))

    gsm=gsm+gsm.transpose()

    for i in np.arange(0,gsm.shape[0]):
        gsm[i,i]=gsm[i,i]/2

    return gsm

def generate_sample_mut_similarity(cancerType):
    mut_P=pd.read_table(filepath_or_buffer='./data/%s/mutation_P.txt' % cancerType,index_col=0,header=0,sep='\t')
    P_orig = pd.read_table(filepath_or_buffer='./data/%s/P_filtered.txt' % cancerType, header=0, index_col=0, sep='\t')
    ssm=calculate_samp_gaussian_similarity(mut_P)
    X_orig=calculate_mut_gaussian_similarity(P_orig)
    np.savetxt('./data/%s/mut_similarity/mut_sim.txt' % cancerType, X_orig, fmt='%f', delimiter='\t')
    np.savetxt('./data/%s/sample_similarity/samp_mut_profile_sim.txt' % cancerType, ssm, fmt='%f', delimiter='\t')

def load_network(file_path):
    ppi = pd.read_table(filepath_or_buffer=file_path, header=None,
                        index_col=None, names=['source', 'target'], sep='\t')
    ppi_nodes = pd.concat([ppi['source'], ppi['target']], ignore_index=True)
    ppi_nodes = pd.DataFrame(ppi_nodes, columns=['nodes']).drop_duplicates()
    ppi_nodes.reset_index(drop=True, inplace=True)
    return ppi,ppi_nodes

#~~~~~~~~~~~~~Step 1：~~~~~~~~~~~~~~~~~~
# Prepare data by selecting samples and mutated genes that both available in the
# original RNA-seq.txt and SomaticMutation.txt downloaded from TCGA.
# Prepared data is saved in the directory: './data/%s/ % cancer_folder'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
prepare_intersection_data(cancerType=cancer_folder)

#~~~~~~~~~~~~~Step 2：~~~~~~~~~~~~~~~~~~
# Create the mutated gene-sample association matrix A', and the driver-sample matrix A
create_mutation_and_driver_matrices(cancerType=cancer_folder)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~Step 3：~~~~~~~~~~~~~~~~~~
# Calculate the Gaussian interaction profile kernel similarity between mutated genes/samples.
# The files are saving in the following paths,
# './data/%s/sample_similarity/mut_sim.txt' % cancer_folder
# './data/%s/sample_similarity/samp_mut_profile_sim.txt' % cancer_folder
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
generate_sample_mut_similarity(cancerType=cancer_folder)

