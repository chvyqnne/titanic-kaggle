from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from models import evaluate_model

def build_stacking_classifier(best_lr, best_rf, best_knn, meta_model=None):
    if meta_model is None:
        meta_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.8
        )

    stack_model = StackingClassifier(
        estimators=[
            ('lr', best_lr),
            ('rf', best_rf),
            ('knn', best_knn)
        ],
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )
    return stack_model

def train_and_evaluate_stacking_model(X_train, y_train, X_val, y_val, stack_model):
    stack_model.fit(X_train, y_train)
    y_pred = stack_model.predict(X_val)
    print("\n--- Stacking Classifier ---")
    evaluate_model(y_val, y_pred)
    return stack_model
