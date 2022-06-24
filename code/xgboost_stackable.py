import xgboost as xgb
from utils import evaluate_reg
from utils import create_train_test_sets
import pandas as pd
import mlflow

model_name = "xgboost"


def get_model():
    xg_reg = xgb.XGBRegressor(
        objective="reg:squarederror",
        colsample_bytree=0.3,
        learning_rate=0.1,
        n_estimators=3000,
        max_depth=4,
        early_stopping_rounds=50,
    )
    return xg_reg


if __name__ == "__main__":
    df = pd.read_parquet(
        "/Users/TimothyW/Fun/cricket_prediction/data/processed_first_innings/first_innings_processed.parquet"
    )

    config = {
        "test_size": 0.2,
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

        X_train, X_test, y_train, y_test = create_train_test_sets(
            df, features, one_hot_columns, target, config
        )

        xg_reg = get_model()

        xg_reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])

        plots, metrics = evaluate_reg(xg_reg, X_test, y_test)

        for _, path in plots.items():
            mlflow.log_artifact(path)

        mlflow.log_metrics(metrics)
