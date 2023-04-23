import numpy as np
from pylab import *
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from causalscbench.data_access.create_dataset import CreateDataset
from causalscbench.data_access.utils.splitting import DatasetSplitter

print("Getting data file path...")
data_directory = "/home/mila/s/siba-smarak.panigrahi/scratch/cbc/data" # NOTE: change this to your data directory
path_k562, path_rpe1 = CreateDataset(data_directory, filter=True).load()

print("Splitting data...")
dataset_splitter = DatasetSplitter(path_k562, 1.0)

# get observational (train) data
# interventions_train contains "non-targeting" only
observational_expression_matrix_train, observational_interventions_train, observational_gene_names = dataset_splitter.get_observational()

# get test data
# interventions_test contains "non-targeting", "excluding" and gene-names
test_expression_matrix_test, interventions_test, test_gene_names = dataset_splitter.get_test_data()

# get full interventional (train) data
full_expression_matrix_train, full_interventions_train, full_gene_names = dataset_splitter.get_interventional()
list_full_interventions_train = [x for x in full_interventions_train]
list_full_gene_names = [x for x in full_gene_names]

num_genes_explore = 4
first_n_genes = full_gene_names[:num_genes_explore]
indexer = []
for i in range(len(list_full_interventions_train)):
    if list_full_interventions_train[i] in first_n_genes:
        indexer.append(True)
    else:
        indexer.append(False)
M = full_expression_matrix_train[indexer, :]

pca = PCA(n_components=2)
M_transformed = pca.fit_transform(M)

color_vector = np.unique(np.asarray(list_full_interventions_train)[indexer], return_inverse=True)[1]
plt.figure()
plt.scatter(M_transformed[:,0], M_transformed[:,1], c=color_vector)
plt.title("Genes clustering")
plt.show()
plt.savefig("Genes_clustering.png")

'''

for i, gene_name in enumerate(full_gene_names):
    indexer = []
    for j in range(len(list_full_interventions_train)):
        if list_full_interventions_train[j] == gene_name:
            indexer.append(True)
        else:
            indexer.append(False)
    breakpoint()
    p = full_expression_matrix_train[indexer, :]

# get partial interventional (train) data
partial_intervention_seed = 0
fraction_partial_intervention = 0.5
partial_expression_matrix_train, partial_interventions_train, partial_gene_names = dataset_splitter.get_partial_interventional(fraction_partial_intervention, partial_intervention_seed)
'''
