# imports
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

def load_iris_dataset():
    """
    Load the Iris dataset.
    Returns: tuple: A tuple containing feature vectors and labels.
    """
    iris = load_iris()
    feature_vectors, labels = iris.data, iris.target
    return feature_vectors, labels

def evaluate_classifier_performance_5_fold(X, y):
    """
    Evaluate different classifiers using 5-fold cross-validation.
    Parameters: X (array-like): Feature vectors. y (array-like): Labels.
    Returns: dict: A dictionary containing evaluation results for each classifier.
    """
    # Define models to evaluate
    models = {
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(probability=True),  # enable probability for ROC AUC
        "Random Forest": RandomForestClassifier(random_state=None),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "K-Nearest Neighbors": KNeighborsClassifier()
    }

    # defines the 5-fold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # defines scoring metrics.
    scoring = {
        'Accuracy': make_scorer(accuracy_score),
        'F1-score': make_scorer(f1_score, average='weighted'),
        'ROC AUC': make_scorer(roc_auc_score, needs_proba=True, average='macro', multi_class='ovr')
    }
    
    results = {}

    # Cross validation using vectorization numpy:
    for name, model in models.items():
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)
        results[name] = {
            'Accuracy': np.mean(cv_results['test_Accuracy']),
            'F1-score': np.mean(cv_results['test_F1-score']),
            'ROC AUC': np.mean(cv_results['test_ROC AUC'])
        }

    return results

def main():
    # Iris dataset
    feature_vectors, labels = load_iris_dataset()
    
    # run 5 fold evaluation
    results = evaluate_classifier_performance_5_fold(feature_vectors, labels)
    
    print("Iris Dataset:")
    print("This Iris dataset consists of 150 samples of iris flowers, each with four features: sepal length, sepal width, petal length, and petal width. The target variable in the dataset represents the species of iris, which encompasses three different classes: Setosa [0], Versicolor [1], and Virginica [2].")

    # print results
    print("\nClassifier\t\t\tAccuracy\tF1-score\tROC AUC")
    for clf_name, clf_result in results.items():
        print(f"{clf_name.ljust(25)}\t{clf_result['Accuracy']:.7f}\t{clf_result['F1-score']:.7f}\t"
              f"{clf_result['ROC AUC']:.7f}")

if __name__ == "__main__":
    main()
