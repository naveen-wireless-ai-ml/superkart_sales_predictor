# for data manipulation
import pandas as pd
# for creating a folder
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn import metrics
# for model training, tuning, and evaluation
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
# Libraries to get different metric scores
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
# for model serialization
import joblib
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("superkart-training-experiment")

api = HfApi()

Xtrain_path = "hf://datasets/naveenaggarwal1989/superkart-sales-data/Xtrain.csv"
Xtest_path = "hf://datasets/naveenaggarwal1989/superkart-sales-data/Xtest.csv"
ytrain_path = "hf://datasets/naveenaggarwal1989/superkart-sales-data/ytrain.csv"
ytest_path = "hf://datasets/naveenaggarwal1989/superkart-sales-data/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# List of categorical features in the dataset (ordinal encoding)
ordinal_features = [
    'Product_Sugar_Content',
    'Store_Size',
    'Store_Location_City_Type'
]

# List of categorical features in the dataset (one-hot encoding)
categorical_features = [
    'Product_Type',
    'Store_Type'
]

# List of numerical features in the dataset (excluding 'id' as it is an identifier)
numeric_features = [
    'Product_Weight',
    'Product_Allocated_Area',
    'Product_MRP',
    'Store_Establishment_Year'
]

prod_sugar_content = ['No Sugar', 'Low Sugar', 'Regular']
store_sz = ['Small', 'Medium', 'High']
store_loc_city_type = ['Tier 3', 'Tier 2', 'Tier 1']

ordinal_categories = [
    prod_sugar_content,
    store_sz,
    store_loc_city_type
]

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features),
    (OrdinalEncoder(categories=ordinal_categories, handle_unknown='use_encoded_value', unknown_value=-1), ordinal_features)
)

# Define base XGBoost model
xgb_model = XGBRegressor(random_state=10)

# Define hyperparameter grid
param_grid = {
    'xgbregressor__n_estimators': [50, 75, 100, 150, 200],  # number of trees to build
    'xgbregressor__max_depth': [2, 3, 4, 5],                # maximum depth of each tree
    'xgbregressor__colsample_bytree': [0.4, 0.5, 0.6],      # percentage of attributes to be considered (randomly) for each tree
    'xgbregressor__colsample_bylevel': [0.4, 0.5, 0.6],     # percentage of attributes to be considered (randomly) for each level of a tree
    'xgbregressor__learning_rate': [0.01, 0.05, 0.1],       # learning rate
    'xgbregressor__reg_lambda': [0.4, 0.5, 0.6],            # L2 regularization factor
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.r2_score)

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring=scorer, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)

    # Store and evaluate the best model
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    # Metrics
    train_rmse = mean_squared_error(ytrain, y_pred_train)
    test_rmse = mean_squared_error(ytest, y_pred_test)

    train_mae = mean_absolute_error(ytrain, y_pred_train)
    test_mae = mean_absolute_error(ytest, y_pred_test)

    train_r2 = r2_score(ytrain, y_pred_train)
    test_r2 = r2_score(ytest, y_pred_test)

    # Log metrics
    mlflow.log_metrics({
        "train_RMSE": train_rmse,
        "test_RMSE": test_rmse,
        "train_MAE": train_mae,
        "test_MAE": test_mae,
        "train_R2": train_r2,
        "test_R2": test_r2
    })

    # Save the model locally
    model_path = "best_superkart_sales_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "naveenaggarwal1989/superkart-sales-model"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_superkart_sales_model_v1.joblib",
        path_in_repo="best_superkart_sales_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
