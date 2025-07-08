import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, make_classification
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Function to compute separability measure (trace(Sb)/trace(Sw))
def separability(X, y):
    overall_mean = np.mean(X, axis=0)
    classes = np.unique(y)
    Sb = np.zeros((X.shape[1], X.shape[1]))
    Sw = np.zeros((X.shape[1], X.shape[1]))
    for cls in classes:
        Xc = X[y == cls]
        mean_c = np.mean(Xc, axis=0)
        Sb += len(Xc) * np.outer(mean_c - overall_mean, mean_c - overall_mean)
        Sw += np.cov(Xc, rowvar=False) * (len(Xc) - 1)
    return np.trace(Sb) / np.trace(Sw)

# Generate synthetic binary dataset
X_syn, y_syn = make_classification(
    n_samples=300,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42
)

# Load datasets
datasets = {
    "Breast Cancer": load_breast_cancer(return_X_y=True),
    "Synthetic Binary": (X_syn, y_syn),
    "Iris": load_iris(return_X_y=True),
    "Wine": load_wine(return_X_y=True),
}

# PCA component settings
component_candidates = [2, 5, 10, 20, 30]

for name, (X, y) in datasets.items():
    print(f"\n=== Dataset: {name} ===")
    
    # Task 1: LDA
    classes = np.unique(y)
    n_classes = len(classes)
    n_components_lda = 1 if n_classes == 2 else min(n_classes - 1, 2)
    
    sep_before = separability(X, y)
    lda = LDA(n_components=n_components_lda)
    X_lda = lda.fit_transform(X, y)
    sep_after = separability(X_lda, y)
    print(f"LDA separability BEFORE: {sep_before:.3f}")
    print(f"LDA separability AFTER:  {sep_after:.3f}")
    
    if n_classes == 2:
        y_scores = X_lda.ravel()
        fpr, tpr, _ = roc_curve(y, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.title(f'{name} - LDA ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()
    
    # Task 2: PCA + Logistic Regression only
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print("\nPCA + Logistic Regression results:")
    print("n_comp | Variance_Ratio | LR_Acc")
    
    for n in component_candidates:
        if n > X.shape[1]:
            continue
        pca = PCA(n_components=n)
        X_tr_pca = pca.fit_transform(X_train)
        X_te_pca = pca.transform(X_test)
        lr = LogisticRegression(max_iter=2000, random_state=42)
        lr.fit(X_tr_pca, y_train)
        acc_lr = accuracy_score(y_test, lr.predict(X_te_pca))
        var_ratio = pca.explained_variance_ratio_.sum()
        
        print(f"{n:<6} | {var_ratio:.3f}          | {acc_lr:.3f}")
        
    

