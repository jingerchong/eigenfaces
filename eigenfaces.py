import numpy as np
from sklearn.datasets import fetch_olivetti_faces, fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def get_dataset(name='olivetti'):
    if name == 'olivetti':
        # Load the Olivetti Faces Dataset
        faces_data = fetch_olivetti_faces()
        X = faces_data.data
        y = faces_data.target
        shape = (64, 64)
    elif name == 'lfw':
        X, y = fetch_lfw_people(min_faces_per_person=5, return_X_y=True)
        unique, counts = np.unique(y, return_counts=True)

        Q1 = np.percentile(counts, 25)
        Q3 = np.percentile(counts, 75)
        IQR = Q3 - Q1
        multiplier = 1.5
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        outliers = [person_id for person_id, num_faces in zip(unique, counts)
                    if num_faces > upper_bound or num_faces < lower_bound]
        outlier_mask = np.isin(y, outliers, invert=True)
        X = X[outlier_mask]
        y = y[outlier_mask]
        shape = (62, 47)
    else:
        raise ValueError('invalid dataset name:', name)
    return X, y, shape

def split_dataset(X, y, test_size=0.2, stratify=True):
    # Split the dataset into training and testing sets
    stratify_var = y if stratify else None # If True, makes sure all faces are represented in both datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=stratify_var)
    return X_train, X_test, y_train, y_test

def eigenfaces_pca(X_train, X_test, n_components):
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(X_train)

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca

def test_model(X_train_pca, X_test_pca, y_train, method='norm 2'):
    # Test the model
    type, param = method.split()
    try:
        if type == 'norm':
            y_pred = []
            for i in range(len(X_test_pca)):
                distances = np.linalg.norm(X_train_pca - X_test_pca[i], axis=1, ord=int(param))
                min_distance_idx = np.argmin(distances)
                y_pred.append(y_train[min_distance_idx])

        elif type == 'knn':
            knn = KNeighborsClassifier(n_neighbors=int(param))
            knn.fit(X_train_pca, y_train)
            y_pred = knn.predict(X_test_pca)

        elif type == 'svm':
            # Fit SVM model
            svm = SVC(kernel=param)  # You can try different kernels like 'rbf', 'poly', etc.
            svm.fit(X_train_pca, y_train)
            y_pred = svm.predict(X_test_pca)

        else:
            raise ValueError(f"invalid method {method:s}")
        
    except ValueError:
        raise ValueError(f"invalid param {param:s}")
    
    return y_pred

def get_accuracy(y_test, y_pred, print_flag=False):
    accuracy = accuracy_score(y_test, y_pred)
    if print_flag:
        print(f"Accuracy: {accuracy*100:.2f}%")
    return accuracy

def get_cm(y_test, y_pred, print_flag=False):
    cm = confusion_matrix(y_test, y_pred)
    if print_flag:
        print(f"Accuracy: {cm*100:.2f}%")
    return cm
