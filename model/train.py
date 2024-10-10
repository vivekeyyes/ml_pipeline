import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import subprocess

# Pull dataset from DVC
subprocess.run(["dvc", "pull"], check=True)

# Load dataset
data = pd.read_csv('data/data.csv')
X = data.drop(columns=['target'])
y = data['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow experiment
mlflow.set_experiment("simple_classifier_experiment")

with mlflow.start_run():
    # Train a RandomForest model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics and model to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(clf, "random_forest_model")

    # Save the accuracy for the test step
    with open("model/accuracy.txt", "w") as f:
        f.write(str(accuracy))
