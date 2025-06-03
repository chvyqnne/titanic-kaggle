from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

def grid_search_logistic_regression(X, y):
    param_grid = {
        'C': [0.01, 0.1, 1],
        'solver': ['liblinear', 'lbfgs'],
        'class_weight': [None, 'balanced']
    }
    model = LogisticRegression(max_iter=1000)
    return run_grid_search(model, param_grid, X, y, "Logistic Regression")

def grid_search_random_forest(X, y):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5]
    }
    model = RandomForestClassifier(random_state=42)
    return run_grid_search(model, param_grid, X, y, "Random Forest")

def grid_search_gradient_boosting(X, y):
    param_grid = {
        'n_estimators': [100],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0]
    }
    model = GradientBoostingClassifier()
    return run_grid_search(model, param_grid, X, y, "Gradient Boosting")

def grid_search_knn(X, y):
    param_grid = {
        'n_neighbors': [3, 5, 9],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # Manhattan vs Euclidean
    }
    model = KNeighborsClassifier()
    return run_grid_search(model, param_grid, X, y, "KNN")

def run_grid_search(model, param_grid, X, y, name):
    grid = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X, y)
    print(f"\n--- Grid Search: {name} ---")
    print("Best Parameters:", grid.best_params_)
    print(f"Best F1 Score (CV): {grid.best_score_:.4f}")
    return grid.best_estimator_

def evaluate_model(y_true, y_pred):
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
