import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, auc

# Split data into training and testing sets
def train_test_split(X, y, test_size=0.3, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * (1 - test_size))
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# Gaussian Naïve Bayes
class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.parameters = {}
        for c in self.classes:
            X_c = X[y == c]
            self.parameters[c] = {
                'mean': np.mean(X_c, axis=0),
                'var': np.var(X_c, axis=0),
                'prior': X_c.shape[0] / X.shape[0]
            }
    
    def predict(self, X):
        preds = []
        # Store the log probs 
        log_probs = []
        for x in X:
            posteriors = []
            for c in self.classes:
                prior = np.log(self.parameters[c]['prior'])
                # Calculate likelihood using Gaussian distribution
                var = self.parameters[c]['var']
                mean = self.parameters[c]['mean']
                log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var))
                log_likelihood -= 0.5 * np.sum(((x - mean) ** 2) / var)
                posterior = prior + log_likelihood
                posteriors.append(posterior)
            log_probs.append(posteriors)
            preds.append(self.classes[np.argmax(posteriors)])
        return np.array(preds), np.array(log_probs)

# k-NN classifier
class KNN:
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def predict(self, X):
        preds = []
        scores = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            # Determine the most common class label
            counts = np.bincount(k_nearest_labels)
            pred = np.argmax(counts)
            preds.append(pred)
            # Calculate the score as the proportion of positive labels
            scores.append(np.mean(k_nearest_labels == 1))
        return np.array(preds), np.array(scores)

# evaluation module to calculate confusion matrix and plot it
def confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for i, true in enumerate(classes):
        for j, pred in enumerate(classes):
            matrix[i, j] = np.sum((y_true == true) & (y_pred == pred))
    return matrix

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# For binary classification, plot ROC curve and calculate AUC
def plot_roc_auc(y_true, scores, title='ROC Curve'):
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    return roc_auc

# main program to choose datasets and run classifiers
def main():
    # Choose datasets
    # Binary dataset 1 : Breast Cancer
    bc = datasets.load_breast_cancer()
    X_bc, y_bc = bc.data, bc.target

    # Binary dataset 2 : Synthetic
    X_syn, y_syn = make_classification(n_samples=300, n_features=20,
                                       n_informative=15, n_redundant=5,
                                       n_classes=2, random_state=42)
    
    # Multiclass dataset 1 : Iris
    iris = datasets.load_iris()
    X_iris, y_iris = iris.data, iris.target

    # Multiclass dataset 2 : Wine
    wine = datasets.load_wine()
    X_wine, y_wine = wine.data, wine.target

    datasets_list = [
        ('Breast Cancer', X_bc, y_bc, 'binary'),
        ('Synthetic Binary', X_syn, y_syn, 'binary'),
        ('Iris', X_iris, y_iris, 'multiclass'),
        ('Wine', X_wine, y_wine, 'multiclass')
    ]

    for name, X, y, dtype in datasets_list:
        print(f"\nDataset: {name}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

        # GNB classifier
        print("Training Gaussian Naïve Bayes...")
        gnb = GaussianNaiveBayes()
        gnb.fit(X_train, y_train)
        preds_gnb, log_probs_gnb = gnb.predict(X_test)
        cm_gnb = confusion_matrix(y_test, preds_gnb)
        print("Confusion Matrix (GNB):\n", cm_gnb)
        plot_confusion_matrix(cm_gnb, classes=np.unique(y), title=f'{name} - GNB Confusion Matrix')
        
        if dtype == 'binary':
            pos_idx = list(gnb.classes).index(1)
            scores_gnb = np.exp(log_probs_gnb[:, pos_idx])
            auc_value = plot_roc_auc(y_test, scores_gnb, title=f'{name} - GNB ROC Curve')
            print("AUC (GNB):", auc_value)
        
        # k-NN classifier
        print("Training k-NN classifier...")
        knn = KNN(k=3)
        knn.fit(X_train, y_train)
        preds_knn, scores_knn = knn.predict(X_test)
        cm_knn = confusion_matrix(y_test, preds_knn)
        print("Confusion Matrix (k-NN):\n", cm_knn)
        plot_confusion_matrix(cm_knn, classes=np.unique(y), title=f'{name} - k-NN Confusion Matrix')
        
        if dtype == 'binary':
            auc_value_knn = plot_roc_auc(y_test, scores_knn, title=f'{name} - k-NN ROC Curve')
            print("AUC (k-NN):", auc_value_knn)
        print("-" * 50)

if __name__ == '__main__':
    main()
