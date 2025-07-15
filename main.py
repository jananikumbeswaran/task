from fastapi import FastAPI
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

app = FastAPI()

@app.get("/predict")
def predict():
    train_df = pd.read_csv('heart - Copy.csv')
    x = train_df.drop('target', axis=1)
    y = train_df['target']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    test_df = pd.read_csv('test_data.csv')
    if 'target' in test_df.columns:
        test_df = test_df.drop('target', axis=1)
    models = {
        "NaiveBayes": GaussianNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    results = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pared_test = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pared_test) * 100
        predictions = model.predict(test_df)
        results[name] = {
            "accuracy": accuracy,
            "predictions": predictions.tolist()
        }

    return results