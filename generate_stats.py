from scipy import stats
import numpy as np
import os

# dataset = 'olivetti'
dataset = 'lfw'

results_dir = 'results'
stats_dir = 'stats'


########################
# Set bool flags here! #
########################

num_pca_components_1 = False
num_pca_components_2 = False 
classification_methods = False 


########################
# Num PCA Components 1 #
########################

if num_pca_components_1:

    pca_dir = 'pca'
    diagram_dir = '1'

    ns_components =  np.load(os.path.join(results_dir, dataset, pca_dir, diagram_dir, 'ns_components.npy'))
    test_sizes =  np.load(os.path.join(results_dir, dataset, pca_dir, diagram_dir, 'test_sizes.npy'))
    accuracy =  np.load(os.path.join(results_dir, dataset, pca_dir, diagram_dir, 'accuracy.npy'))
    
    save_dir = os.path.join(stats_dir, dataset, pca_dir, diagram_dir)
    os.makedirs(save_dir, exist_ok=True)
    result_txt = os.path.join(save_dir, 'ttest.txt')

    with open(result_txt, 'w') as file:
        for j in range(len(ns_components)):
            for i_1 in range(len(test_sizes)):
                for i_2 in range(i_1):
                    t_statistic, p_value = stats.ttest_ind(accuracy[i_1, j, :], accuracy[i_2, j, :])
                    # Use the p-value to determine significance
                    if p_value >= 0.05:
                        file.write(f"Test sizes {test_sizes[i_1]:.1f}, {test_sizes[i_2]:.1f} have NO significant difference for n_components={ns_components[j]}\n")
    

########################
# Num PCA Components 2 #
########################

if num_pca_components_2:

    pca_dir = 'pca'
    diagram_dir = '2'

    ns_components =  np.load(os.path.join(results_dir, dataset, pca_dir, diagram_dir, 'ns_components.npy'))
    test_sizes =  np.load(os.path.join(results_dir, dataset, pca_dir, diagram_dir, 'test_sizes.npy'))
    accuracy =  np.load(os.path.join(results_dir, dataset, pca_dir, diagram_dir, 'accuracy.npy'))

    save_dir = os.path.join(stats_dir, dataset, pca_dir, diagram_dir)
    os.makedirs(save_dir, exist_ok=True)
    result_txt = os.path.join(save_dir, 'ttest.txt')

    with open(result_txt, 'w') as file:
        for i in range(len(test_sizes)):
            max_j = np.argmax(np.mean(accuracy[i], axis=1))
            for j in range(len(ns_components)):
                if j != max_j:
                    t_statistic, p_value = stats.ttest_ind(accuracy[i, max_j, :], accuracy[i, j, :])
                    # Use the p-value to determine significance
                    if p_value >= 0.05:
                        file.write(f"Num components {ns_components[max_j]:d}, {ns_components[j]:d} have NO significant difference for test_size={test_sizes[i]:.1f}\n")


##########################
# Classification Methods #
##########################

if classification_methods:

    classify_dir = 'classify'
    methods = np.load(os.path.join(results_dir, dataset, classify_dir, 'methods.npy'))
    accuracy = np.load(os.path.join(results_dir, dataset, classify_dir, 'accuracy.npy'))
    runtime = np.load(os.path.join(results_dir, dataset, classify_dir, 'runtime.npy'))
    
    save_dir = os.path.join(stats_dir, dataset, classify_dir)
    os.makedirs(save_dir, exist_ok=True)
    result_txt = os.path.join(save_dir, 'ttest.txt')

    with open(result_txt, 'w') as file:

        diff_i = []
        max_i = np.argmax(np.mean(accuracy, axis=1))
        file.write(f"Method {methods[max_i]} has highest accuracy\n")

        for i in range(len(methods)):
            if i != max_i:
                t_statistic, p_value = stats.ttest_ind(accuracy[max_i, :], accuracy[i, :])
                # Use the p-value to determine significance
                if p_value >= 0.05:
                    diff_i.append(i)
                    file.write(f"Methods {methods[max_i]}, {methods[i]} have NO significant difference in accuracy\n")

        for i in diff_i:
            t_statistic, p_value = stats.ttest_ind(runtime[max_i, :], runtime[i, :])
            file.write(f"Methods {methods[max_i]}, {methods[i]} have significant difference in runtime\n")

