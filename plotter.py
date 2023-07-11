import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics


def corr_heatmap(X_train):
    X_train.corr()
    sns.heatmap(X_train.corr())


def roc_curve_plot(y_val, y_pred, title):
    fpr, tpr, _ = metrics.roc_curve(y_val, y_pred)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('RoC Curve - ' + title)
    plt.show()
