import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
import mlflow


def determine_bowl_team(row):
    if row["batting_team"] == row["team_one"]:
        return row["team_two"]
    return row["team_one"]


def rmse_at_balls(test_df, balls):
    ball_df = test_df[test_df.overall_ball_n == balls]
    return np.sqrt(mean_squared_error(ball_df["actual"], ball_df["preds"]))


def rmse_team(test_df, country):
    country_df = test_df[test_df.batting_team == country]
    return np.sqrt(mean_squared_error(country_df["actual"], country_df["preds"]))


def plot_residuals(residuals, bins=40):
    fig = plt.figure()
    ax = fig.gca()
    ax.cla()
    ax.hist(residuals, bins=bins)
    return fig


def plot_rmses(x, y):
    fig = plt.figure()
    ax = fig.gca()
    ax.cla()
    ax.plot(x, y)
    ax.grid(visible=True)
    return fig


if __name__ == "__main__":
    df = pd.read_parquet(
        "/Users/TimothyW/Fun/cricket_prediction/data/processed_first_innings/first_innings_processed.parquet"
    )

    df = df[
        ~((df.total_balls_bowled < 300) & (df.total_wickets_lost < 10))
    ]  # remove rain delays

    df["batter_won_toss"] = df["batting_team"] == df["toss_winner"]

    df["bowling_team"] = df.apply(determine_bowl_team, axis=1)

    df["year"] = df["date"].str[0:4]

    df["match_id"] = df["city"] + df["batting_team"] + df["bowling_team"] + df["date"]

    df["runs_per_ball"] = df["current_runs"] / df["overall_ball_n"]

    df = df[~df.city.isna()]

    with mlflow.start_run():

        features = [
            "current_runs",
            "current_wickets",
            "overall_ball_n",
            "batting_team",
            "batter_won_toss",
        ]
        target = ["first_innings_total_runs"]

        config = {
            "test_size": 0.2,
            "learning_rate": 0.03,
            "iterations": 3000,
            "eval_metric": "RMSE",
            "early_stopping_rounds": 50,
            "l2_leaf_reg": 2,
            "max_depth": 3,
        }

        mlflow.log_param("features", features)

        mlflow.log_params(config)

        # train test split

        splitter = GroupShuffleSplit(
            test_size=config["test_size"], n_splits=2, random_state=7
        )
        split = splitter.split(df, groups=df["match_id"])
        train_inds, test_inds = next(split)
        train = df.iloc[train_inds]
        test = df.iloc[test_inds]
        X_train = train[features]
        X_test = test[features]
        y_train = train[target]
        y_test = test[target]

        clf = CatBoostRegressor(
            learning_rate=config["learning_rate"],
            iterations=config["iterations"],
            eval_metric=config["eval_metric"],
            l2_leaf_reg=config["l2_leaf_reg"],
            max_depth=config["max_depth"],
        )

        cat_feature_ind = [3, 4]
        clf.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            early_stopping_rounds=config["early_stopping_rounds"],
            cat_features=cat_feature_ind,
        )

        mlflow.log_param("cat_feature_indices", cat_feature_ind)

        preds = clf.predict(X_test)

        flat_y_test = y_test.values.flatten()

        residuals = preds - flat_y_test

        plots_path = "/Users/TimothyW/Fun/cricket_prediction/plots/"

        resid_fig = plot_residuals(residuals)
        resid_fig.savefig(f"{plots_path}/residuals_hist.png")
        mlflow.log_artifact(f"{plots_path}/residuals_hist.png")

        X_test["preds"] = preds
        X_test["actual"] = y_test
        X_test["residuals"] = preds - flat_y_test

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mlflow.log_metric("RMSE", rmse)

        mid_way = X_test[X_test.overall_ball_n == 150]

        rmse_halfway = np.sqrt(mean_squared_error(mid_way["actual"], mid_way["preds"]))
        mlflow.log_metric("rmse_halfway", rmse_halfway)

        key_balls = [30, 180, 240, 270]
        for kb in key_balls:
            rmse_kb = rmse_at_balls(X_test, kb)
            mlflow.log_metric(f"rmse_at_{kb}_balls", rmse_kb)

        rmse_x = []
        rmse_y = []
        for i in range(12, 270):
            rmse_x.append(i)
            rmse_y.append(rmse_at_balls(X_test, i))

        rmse_plot = plot_rmses(rmse_x, rmse_y)
        rmse_plot.savefig(f"{plots_path}/rmse_plot.png")
        mlflow.log_artifact(f"{plots_path}/rmse_plot.png")
