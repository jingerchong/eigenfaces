from eigenfaces import *
import os, time

# dataset = 'olivetti'
# test_size = 0.2
# n_components = 21

dataset = 'lfw'
test_size = 0.2
n_components = 54

n_trials = 20
X, y, _ = get_dataset(dataset)

methods = ['norm 1', 'norm 2', 'knn 3', 'knn 5', 'knn 7', 'svm linear', 'svm rbf', 'svm poly']

accuracy = np.zeros((len(methods), n_trials))
runtime = np.zeros_like(accuracy)

for k in range(n_trials):

    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size)
    X_train_pca, X_test_pca = eigenfaces_pca(X_train, X_test, n_components)

    for i in range(len(methods)):
        
        start_time = time.time()
        y_pred = test_model(X_train_pca, X_test_pca, y_train, methods[i])
        runtime[i, k] = time.time() - start_time
        accuracy[i, k] = get_accuracy(y_test, y_pred)

save_dir = os.path.join('results', dataset, os.path.splitext(os.path.basename(__file__))[0])
os.makedirs(save_dir, exist_ok=True)
np.save(os.path.join(save_dir, 'methods.npy'), methods)
np.save(os.path.join(save_dir, 'accuracy.npy'), accuracy)
np.save(os.path.join(save_dir, 'runtime.npy'), runtime)
