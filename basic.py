from eigenfaces import *
import os
import numpy as np
import matplotlib.pyplot as plt

# dataset = 'olivetti'
dataset = 'lfw'

n_diagrams = 3
X, y, shape = get_dataset(dataset)

X_train, X_test, y_train, y_test = split_dataset(X, y)
X_train_pca, X_test_pca = eigenfaces_pca(X_train, X_test, len(np.unique(y))//2)
y_pred = test_model(X_train_pca, X_test_pca, y_train)

for i in range(n_diagrams):

    idx = np.random.randint(len(X_test))
    actual_label = y_test[idx]
    actual_img = X_test[idx].reshape(shape)
    predicted_label = y_pred[idx]
    predicted_face_idx = np.where(y_train == predicted_label)[0][0]
    predicted_img = X_train[predicted_face_idx].reshape(shape)

    diagram_str = str(i+1)
    save_dir = os.path.join('results', dataset, os.path.splitext(os.path.basename(__file__))[0], diagram_str)
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'labels.npy'), [actual_label, predicted_label])
    np.save(os.path.join(save_dir, 'actual_img.npy'), actual_img)
    np.save(os.path.join(save_dir, 'predicted_img.npy'), predicted_img)

print(f'Accuracy: {get_accuracy(y_test, y_pred):.2f}')

# olivetti 
# Accuracy: 0.89

# lfw 
# Accuracy: 0.14