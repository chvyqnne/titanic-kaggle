import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from eda import preprocess_data
from stacking import build_stacking_classifier, train_and_evaluate_stacking_model

def get_final_models():
    lr = LogisticRegression(C=0.1, class_weight='balanced', solver='liblinear', max_iter=1000)
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, random_state=42)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8)
    knn = KNeighborsClassifier(n_neighbors=9, weights='distance', p=1)
    return lr, rf, gb, knn

def load_and_preprocess_data():
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')

    train_data = preprocess_data(train_df)

    return train_df, test_df, train_data

def scale_features(train_data, test_data):
    X = train_data.drop(columns=['Survived'])
    y = train_data['Survived']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    test_scaled = scaler.transform(test_data)

    return X, y, X_scaled, test_scaled

def train_stack_model(X_scaled, y):
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    lr, rf, gb, knn = get_final_models()
    stack_model = build_stacking_classifier(lr, rf, knn, meta_model=gb)
    return train_and_evaluate_stacking_model(X_train, y_train, X_val, y_val, stack_model)

def generate_submission(model, X_train, y_train, X_test, passenger_ids, output_path='outputs/submission.csv'):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    submission = pd.DataFrame({
        'PassengerId': passenger_ids,
        'Survived': predictions
    })
    submission.to_csv(output_path, index=False)

def main():
    _, test_df, train_data = load_and_preprocess_data()
    
    test_data = preprocess_data(test_df).fillna(test_df.median(numeric_only=True))
    
    _, y, X_scaled, test_scaled = scale_features(train_data, test_data)

    trained_stack = train_stack_model(X_scaled, y)

    generate_submission(trained_stack, X_scaled, y, test_scaled, test_df['PassengerId'])


if __name__ == '__main__':
    main()
