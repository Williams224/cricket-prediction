from sklearn import metrics
from catboost_model import catboost_model
from xgboost_model import xgboost_model
from sklearn.linear_model import LinearRegression
import mlflow
import utils
import pandas as pd
from sklearn.model_selection import GroupKFold

if __name__ == "__main__":

    df = pd.read_parquet(
        "/Users/TimothyW/Fun/cricket_prediction/data/processed_first_innings/first_innings_processed.parquet"
    ).reset_index()

    df = df[~df.city.isna()]

    df = df[df.overall_ball_n > 0]

    df = df[
        ~((df.total_balls_bowled < 300) & (df.total_wickets_lost < 10))
    ]  # remove rain delays

    df = df[df.batting_team_balls_faced > 3000]

    df["current_run_rate"] = df["current_runs"] / df["overall_ball_n"]

    df["naive_projection"] = (
        df["current_run_rate"] * df["innings_balls_left"] + df["current_runs"]
    )

    df["mean_encoded_batting_team"] = df.groupby("batting_team")[
        "first_innings_total_runs"
    ].transform("mean")

    target_name = "first_innings_total_runs"

    gkf = GroupKFold(2)

    prediction_dfs = []

    mlflow.set_experiment("stacked")
    with mlflow.start_run():

        if True:
            for train_index, test_index in gkf.split(df, groups=df["match_id"]):
                df_test = df.iloc[test_index]
                # =================================== catboost =============================================
                catboost_model.fit(
                    df,
                    train_index,
                    test_index,
                    target_name,
                    {"early_stopping_rounds": 50},
                )
                print(" catboost fit done")
                catboost_predictions = catboost_model.predict(df_test)
                df_test[f"{catboost_model.name}_predictions"] = catboost_predictions
                print(" catboost predictions done")

                (
                    catboost_fi_plot_path,
                    catboost_fi_json_path,
                ) = utils.get_feature_importances(catboost_model, df_test, target_name)
                mlflow.log_artifact(catboost_fi_plot_path)
                mlflow.log_artifact(catboost_fi_json_path)

                print("catboost feature importance done")
                # =================================== xgboost =============================================
                xgboost_model.fit(df, train_index, test_index, target_name, {})
                print("xgboost fit done")
                xgboost_predictions = xgboost_model.predict(df_test)
                df_test[f"{xgboost_model.name}_predictions"] = xgboost_predictions
                print("xgboost predictions done")

                (
                    xgboost_fi_plot_path,
                    xgboost_fi_json_path,
                ) = utils.get_feature_importances(xgboost_model, df_test, target_name)
                mlflow.log_artifact(xgboost_fi_plot_path)
                mlflow.log_artifact(xgboost_fi_json_path)

                prediction_dfs.append(df_test)

            df_all_preds = pd.concat(prediction_dfs)

            catboost_plots, catboost_metrics = utils.evaluate_reg(
                df_all_preds, catboost_model.name, target_name
            )

            xgboost_plots, xgboost_metrics = utils.evaluate_reg(
                df_all_preds, xgboost_model.name, target_name
            )

            metrics = {**catboost_metrics, **xgboost_metrics}
            plots = {**catboost_plots, **xgboost_plots}

            mlflow.log_metrics(metrics)

            for _, v in plots.items():
                mlflow.log_artifact(v)
            print("all done")

            df_all_preds.to_parquet(
                "/Users/TimothyW/Fun/cricket_prediction/data/modelling_results/before_meta.parquet"
            )
        else:
            df_all_preds = pd.read_parquet(
                "/Users/TimothyW/Fun/cricket_prediction/data/modelling_results/before_meta.parquet"
            )

        # ==================================== meta regressor ================================================

        meta_features = [
            "xgboost_predictions",
            "catboost_predictions",
        ]
        meta_gkf = GroupKFold(2)

        meta_preds = []
        index = 0
        for train_index, test_index in meta_gkf.split(
            df_all_preds, groups=df_all_preds[["match_id"]]
        ):
            df_test = df_all_preds.iloc[test_index]
            df_train = df_all_preds.iloc[train_index]

            X_train = df_train[meta_features]
            X_test = df_test[meta_features]
            y_train = df_train[[target_name]]
            y_test = df_test[[target_name]]

            reg = LinearRegression().fit(X_train, y_train)

            score = reg.score(X_train, y_train)
            mlflow.log_metric(f"score_{index}", score)
            predictions = reg.predict(X_test)
            print(reg.coef_)
            print(reg.intercept_)

            df_test["meta_predictions"] = predictions
            meta_preds.append(df_test)
            index += 1

        df_meta_preds = pd.concat(meta_preds, axis=0)

        meta_plots, meta_metrics = utils.evaluate_reg(
            df_meta_preds, "meta", target_name
        )

        mlflow.log_metrics(meta_metrics)

        for _, v in meta_plots.items():
            mlflow.log_artifact(v)

    # model_a.fit(df, train_index, target_name)
    # df_test
    # predictions = model_a.predict(df_test)
    # metrics_dict, plots_dict = evaluate(model_a, df_test, predictions)
    # df_test[predictions] = predictions
    # preds.append(df_test)

    # mlflow.log_shit

    # concat_predictions

    # shuffle groups around

    # meta_gkf = GroupKFold(2)
    # for train_index, test_index in meta_gkf.split(df, groups=df["match_id"]):
    # meta_model.fit()
    # meta_model.predict()

    # concat_predictions.
    # evaluate_overall.
