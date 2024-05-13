from eigenfaces import *
import os
import time

# dataset = 'olivetti'
# test_size = 0.2
# n_components = 21
# method = 'norm 1'

dataset = 'lfw'
test_size = 0.2
n_components = 54
method = 'svm linear'

n_trials = 20
X, y, _ = get_dataset(dataset)

accuracy = np.zeros(n_trials)
runtime = np.zeros_like(accuracy)
num_faces = len(np.unique(y))
ave_cm = np.zeros((num_faces, num_faces))

for i in range(n_trials):
    
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size)
    X_train_pca, X_test_pca = eigenfaces_pca(X_train, X_test, n_components)
    start_time = time.time()
    y_pred = test_model(X_train_pca, X_test_pca, y_train, method)
    runtime[i] = time.time() - start_time
    accuracy[i] = get_accuracy(y_test, y_pred)
    ave_cm += confusion_matrix(y_test, y_pred)

save_dir = os.path.join('results', dataset, os.path.splitext(os.path.basename(__file__))[0])
os.makedirs(save_dir, exist_ok=True)
np.save(os.path.join(save_dir, 'ave_cm.npy'), ave_cm)

print("Accuracy mean:", np.mean(accuracy), "std:", np.std(accuracy))
print("Train+test runtime mean:", np.mean(runtime), "std:", np.std(runtime))

# olivetti
# Accuracy mean: 0.9531249999999998 std: 0.021605482521804498
# Train+test runtime mean: 0.09861487150192261 std: 0.07868128356798337

# lfw
# Accuracy mean: 0.26503597122302164 std: 0.010608522930167919
# Train+test runtime mean: 5.155400395393372 std: 0.5372842090985488