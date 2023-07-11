
def fit_predict(classifier, X_train, X_validate, y_train, X_test):
    classifier.fit(X_train, y_train)
    y_pred_valid = classifier.predict(X_validate)
    y_pred_test = classifier.predict(X_test)

    return y_pred_valid, y_pred_test
