import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

# dataset = 'olivetti'
dataset = 'lfw'

results_dir = 'results'
diagrams_dir = 'diagrams'


########################
# Set bool flags here! #
########################

visualize_prediction = False
num_pca_components = False
classification_methods = False
confusion_matrix = False


########################
# Visualize Prediction #
########################

if visualize_prediction: 

    basic_dir = 'basic'

    for f in os.listdir(os.path.join(results_dir, dataset, basic_dir)):

        actual_label, predicted_label = np.load(os.path.join(results_dir, dataset, basic_dir, f, 'labels.npy'))
        actual_img = np.load(os.path.join(results_dir, dataset, basic_dir, f, 'actual_img.npy'))
        predicted_img = np.load(os.path.join(results_dir, dataset, basic_dir, f, 'predicted_img.npy'))

        plt.figure(figsize=(8, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(actual_img, cmap='gray')
        plt.title(f"Actual Label: {actual_label:d}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(predicted_img, cmap='gray')
        plt.title(f"Predicted Label: {predicted_label:d}")
        plt.axis('off')

        plt.savefig(os.path.join(diagrams_dir, f'{dataset}_basic_{f}.png'))


######################
# Num PCA Components #
######################

if num_pca_components:

    pca_dir = 'pca'

    for f in os.listdir(os.path.join(results_dir, dataset, pca_dir)):
        
        ns_components =  np.load(os.path.join(results_dir, dataset, pca_dir, f, 'ns_components.npy'))
        test_sizes =  np.load(os.path.join(results_dir, dataset, pca_dir, f, 'test_sizes.npy'))
        accuracy =  np.load(os.path.join(results_dir, dataset, pca_dir, f, 'accuracy.npy'))

        plt.figure(figsize=(8, 6))

        for i in range(len(test_sizes)):
            plt.errorbar(ns_components, np.mean(accuracy[i], axis=1), yerr=np.std(accuracy[i], axis=1),
                    fmt='-',  capsize=2, label=f'test_size {test_sizes[i]:.1f}')

        plt.xlabel('Number of Components')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')

        plt.savefig(os.path.join(diagrams_dir, f'{dataset}_pca_{f}.png'))


##########################
# Classification Methods #
##########################

if classification_methods:

    classify_dir = 'classify'
    methods = np.load(os.path.join(results_dir, dataset, classify_dir, 'methods.npy'))
    accuracy = np.load(os.path.join(results_dir, dataset, classify_dir, 'accuracy.npy'))
    runtime = np.load(os.path.join(results_dir, dataset, classify_dir, 'runtime.npy'))

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.boxplot(accuracy.T, labels=methods, showmeans=True)
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Method')
    plt.savefig(os.path.join(diagrams_dir, f'{dataset}_classify_accuracy.png'))

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.bar(methods, np.mean(runtime, axis=1), yerr=np.std(runtime, axis=1), capsize=5)
    ax2.set_ylabel('Runtime (s)')
    ax2.set_xlabel('Method')
    plt.savefig(os.path.join(diagrams_dir, f'{dataset}_classify_runtime.png'))


####################
# Confusion Matrix #
####################

if confusion_matrix:

    cm_dir = 'best'
    ave_cm = np.load(os.path.join(results_dir, dataset, cm_dir, 'ave_cm.npy'))

    plt.figure(figsize=(8, 6))

    sns.heatmap(ave_cm, cmap='Blues', fmt='d', xticklabels=False, yticklabels=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.savefig(os.path.join(diagrams_dir, f'{dataset}_confusion_matrix.png'))
