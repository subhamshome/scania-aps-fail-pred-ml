from sklearn import metrics
from plotter import roc_curve_plot


def acc_calc(true, pred):
    print("F1: ", metrics.f1_score(true, pred))
    print("Precision: ", metrics.precision_score(true, pred))
    print("Recall: ", metrics.recall_score(true, pred))
    print('Validation F3 Score:', metrics.fbeta_score(true, pred, beta=3))
    print()
    print('Confusion Matrix')
    print(metrics.confusion_matrix(true, pred))
