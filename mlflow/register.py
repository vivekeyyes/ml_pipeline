import mlflow
import mlflow.sklearn

# Get the last run ID
client = mlflow.tracking.MlflowClient()
experiment_id = client.get_experiment_by_name("simple_classifier_experiment").experiment_id
run_id = client.list_run_infos(experiment_id)[-1].run_id

# Register the model
mlflow.register_model(f"runs:/{run_id}/random_forest_model", "RandomForestClassifierModel")
print(f"Model registered successfully from run {run_id}")
