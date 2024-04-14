from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score
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

    # Define custom scorers
    custom_scorers = {
        "Accuracy": make_scorer(accuracy_score),
        "F1-score": make_scorer(f1_score, average='weighted'),
        "ROC AUC": make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr')
    }

    # Perform 5-fold cross-validation for each classifier
    results = {}
    print("Results:")
    print("Classifier\tAccuracy\tF1-score\tROC AUC")
    for clf_name, clf in classifiers.items():
        print(f"Evaluating {clf_name}...")
        clf_results = {}
        for metric_name, scorer in custom_scorers.items():
            scores = cross_val_score(clf, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42),
                                     scoring=scorer)
            clf_results[metric_name] = scores.mean()
        print(f"{clf_name}\t{clf_results['Accuracy']:.7f}\t\t{clf_results['F1-score']:.7f}\t\t"
              f"{clf_results['ROC AUC']:.7f}")

if __name__ == "__main__":
    main()
