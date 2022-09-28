from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

def logistic_regression(random_state=42, solver='saga', max_iter=1000, tol=1e-3):
    print("Fit a logistic regression as meta model... wait a sec")
    return LogisticRegression(random_state=random_state, solver=solver, max_iter=max_iter, tol=tol)

def gradient_boosting(n_estimators=100, min_samples_split=2, min_samples_leaf=1, max_depth=3, random_state=42):
    print("Train a gradient boosting tree as meta model...")
    return GradientBoostingClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split,
                                      min_samples_leaf=min_samples_leaf, max_depth=max_depth, random_state=random_state)
