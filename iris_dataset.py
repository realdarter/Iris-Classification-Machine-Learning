from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

def main():
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Initialize classifiers
    classifiers = {
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(probability=True),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier()
    }

    # Initialize evaluation metrics
    # Initialize evaluation metrics
    metrics = {
        "Accuracy": "accuracy",
        "F1-score_micro": "f1_micro",
        "F1-score_macro": "f1_macro",
        "F1-score_weighted": "f1_weighted",
        "ROC AUC": "roc_auc_ovr"
    }


    # Perform 5-fold cross-validation for each classifier
    results = {}
    for clf_name, clf in classifiers.items():
        print(f"Evaluating {clf_name}...")
        clf_results = {}
        for metric_name, scoring in metrics.items():
            scores = cross_val_score(clf, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                     scoring=scoring)
            clf_results[metric_name] = scores.mean()
        results[clf_name] = clf_results

    # Print results
    print("\nResults:")
    print("Classifier\tAccuracy\tF1-score_micro\tF1-score_macro\tF1-score_weighted\tROC AUC")
    for clf_name, clf_result in results.items():
        print(f"{clf_name}\t{clf_result['Accuracy']:.4f}\t\t{clf_result['F1-score_micro']:.4f}\t\t"
              f"{clf_result['F1-score_macro']:.4f}\t\t{clf_result['F1-score_weighted']:.4f}\t\t"
              f"{clf_result['ROC AUC']:.4f}")


if __name__ == "__main__":
    main()
