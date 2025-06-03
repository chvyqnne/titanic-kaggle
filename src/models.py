import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from tuning import evaluate_model

def select_important_features(model, feature_names, threshold=0.01):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    selected_features = importances[importances >= threshold].index.tolist()
    return selected_features

def voting_ensemble_model(X_train, y_train, X_val, y_val, best_lr, best_rf, best_gb):
    print("\n--- Voting Classifier Ensemble ---")
    ensemble = VotingClassifier(
    estimators=[
        ('lr', best_lr),
        ('rf', best_rf),
        ('gb', best_gb),
    ],
    voting='soft')
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_val)
    evaluate_model(y_val, y_pred)
    return ensemble

def logistic_regression_model(X_train, y_train, X_val, y_val):
    print("\n--- Logistic Regression ---")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    evaluate_model(y_val, y_pred)
    return model

def random_forest_model(X_train, y_train, X_val, y_val):
    print("\n--- Random Forest ---")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    evaluate_model(y_val, y_pred)
    return model

def knn_model(X_train, y_train, X_val, y_val):
    print("\n--- K-Nearest Neighbors ---")
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    evaluate_model(y_val, y_pred)
    return model
    
def gradient_boost_model(X_train, y_train, X_val, y_val):
    print("\n--- Gradient Boosting ---")
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    evaluate_model(y_val, y_pred)
    return model

def cross_validate_model(model, X, y, cv=5):
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    print(f"F1 Score (CV={cv}): {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
    
def cross_validate_all_models(X, y):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'KNN': KNeighborsClassifier(),
        'Gradient Boosting': GradientBoostingClassifier()
    }

    print("\n=== Cross-Validation Results (F1 Score) ===")
    for name, model in models.items():
        if name in ['Logistic Regression', 'KNN']:
            scores = cross_val_score(model, StandardScaler().fit_transform(X), y, cv=5, scoring='f1')
        else:
            scores = cross_val_score(model, X, y, cv=5, scoring='f1')
        print(f"{name:<20}: {scores.mean():.4f} ± {scores.std():.4f}")

def run_all_models(X_train, y_train, X_val, y_val):
    print("\n=== Running All Models ===")
    models = {
        'Logistic Regression': logistic_regression_model,
        'Random Forest': random_forest_model,
        'KNN': knn_model,
        'Gradient Boosting': gradient_boost_model
    }

    trained_models = {}
    for name, func in models.items():
        print(f"\n--- {name} ---")
        trained_models[name] = func(X_train, y_train, X_val, y_val)
    
    return trained_models

