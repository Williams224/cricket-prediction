from nis import cat
from catboost_model import catboost_model

import pandas as pd
from sklearn.model_selection import GroupKFold

if __name__ == "__main__":

    df = pd.read_parquet(
        "/Users/TimothyW/Fun/cricket_prediction/data/processed_first_innings/first_innings_processed.parquet"
    )

    df = df[~df.city.isna()]

    target_name = "first_innings_total_runs"

    gkf = GroupKFold(5)

    for train_index, test_index in gkf.split(df, groups=df["match_id"]):
        catboost_model.fit(
            df, train_index, test_index, target_name, {"early_stopping_rounds": 50}
        )

        print("fit done")

        # model_a.fit(df, train_index, target_name)
        # df_test
        # predictions = model_a.predict(df_test, target_name)
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
