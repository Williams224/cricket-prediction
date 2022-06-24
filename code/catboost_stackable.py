import pandas as pd
from catboost import CatBoostRegressor
import mlflow
from utils import create_train_test_sets, evaluate_reg

from sklearn.model_selection import GroupShuffleSplit

model_name = "catboost"


def get_model():
    clf = CatBoostRegressor(
        learning_rate=0.03,
        iterations=3000,
        eval_metric="RMSE",
        l2_leaf_reg=2,
        max_depth=3,
    )
    return clf


if __name__ == "__main__":
    df = pd.read_parquet(
        "/Users/TimothyW/Fun/cricket_prediction/data/processed_first_innings/first_innings_processed.parquet"
    )

    config = {
        "test_size": 0.5,
        "learning_rate": 0.03,
        "iterations": 3000,
        "eval_metric": "RMSE",
        "early_stopping_rounds": 50,
        "l2_leaf_reg": 2,
        "max_depth": 3,
        "min_bat_team_balls_faced": 4000,
    }

    # find one_hot encoded column names
    one_hot_columns = ["batting_team", "batter_won_toss"]

    features = [
        "current_runs",
        "current_wickets",
        "overall_ball_n",
        "batter_current_runs",
    ]
    target = ["first_innings_total_runs"]

    mlflow.set_experiment(model_name)
    with mlflow.start_run():

        mlflow.log_param("features", features + one_hot_columns)

        mlflow.log_params(config)

        splitter = GroupShuffleSplit(
            test_size=config["test_size"], n_splits=2, random_state=69
        )

        X_a, X_b, y_a, y_b = create_train_test_sets(
            df, features, one_hot_columns, target, splitter, config
        )

        reg_A = get_model()

        # cat_feature_ind = [3, 4]
        reg_A.fit(
            X_a,
            y_a,
            eval_set=(X_b, y_b),
            early_stopping_rounds=config["early_stopping_rounds"],
            #cat_features=cat_feature_ind,
        )

        plots, metrics = evaluate_reg(reg_A, X_b, y_b)

        for _, path in plots.items():
            mlflow.log_artifact(path)

        mlflow.log_metrics(metrics)
