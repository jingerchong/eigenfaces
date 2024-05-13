from eigenfaces import *
import os

# dataset = 'olivetti'
# ns_components_list = [np.array([1, 2, 5, 10, 20, 50, 100, 150]),
#                       np.arange(10, 51)]
# test_sizes_list = [np.arange(0.2, 0.6, 0.1),
#                    np.array([0.2])]

dataset = 'lfw'
ns_components_list = [np.array([1, 2, 5, 10, 20, 50, 100, 250]),
                      np.arange(20, 101)]
test_sizes_list = [np.arange(0.2, 0.6, 0.1),
                   np.array([0.2])]

n_trials = 20
X, y, _ = get_dataset(dataset)

n_diagrams = len(ns_components_list)

for x in range(1, n_diagrams):
    print('diagram', x)

    ns_components = ns_components_list[x]
    test_sizes = test_sizes_list[x]

    accuracy = np.zeros((len(test_sizes), len(ns_components), n_trials))

    for i in range(len(test_sizes)):
        print('test_size', test_sizes[i])

        for j in range(len(ns_components)):
            print('components', ns_components[j])

            for k in range(n_trials):
                X_train, X_test, y_train, y_test = split_dataset(X, y, test_sizes[i])
                X_train_pca, X_test_pca = eigenfaces_pca(X_train, X_test, ns_components[j])
                y_pred = test_model(X_train_pca, X_test_pca, y_train)
                accuracy[i, j, k] = get_accuracy(y_test, y_pred)
        
    diagram_str = str(x+1)
    save_dir = os.path.join('results', dataset, os.path.splitext(os.path.basename(__file__))[0], diagram_str)
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'ns_components.npy'), ns_components)
    np.save(os.path.join(save_dir, 'test_sizes.npy'), test_sizes)
    np.save(os.path.join(save_dir, 'accuracy.npy'), accuracy)

