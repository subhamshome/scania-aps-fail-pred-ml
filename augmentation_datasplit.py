from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE  # type: ignore


def augmentation(X_train_preprocess, X_test_preprocess, y_train):
    pca = PCA(n_components=25)
    X_train_pca = pca.fit(X_train_preprocess).transform(X_train_preprocess)
    X_test_pca = pca.transform(X_test_preprocess)

    X, y = SMOTE(random_state=40).fit_resample(X_train_pca, y_train)

    return X, y, X_test_pca


def datasplit(X, y, X_test_pca):
    X_train1, X_validate, y_train1, y_validate = train_test_split(
        X, y, test_size=0.25, random_state=21, stratify=y)

    y_train1_ravel = y_train1.values.ravel()
    y_validate_ravel = y_validate.values.ravel()

    return X_train1, X_validate, y_train1, y_validate, y_train1_ravel, y_validate_ravel, X_test_pca
