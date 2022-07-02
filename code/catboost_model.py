from base_model import BaseModel
from collections import namedtuple
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold

from sklearn.inspection import permutation_importance


class CatboostModel(BaseModel):
    def __init__(self, name, features, model_params):
        super().__init__(name, features)
        self.model = CatBoostRegressor(**model_params)
        self.cat_feature_indices = [
            self.all_features.index(cf) for cf in self.cat_features
        ]

    def fit(self, data, train_indices, eval_indices, target_name, fit_params):
        train = data.iloc[train_indices]
        eval = data.iloc[eval_indices]
        X_train = train[self.all_features]
        y_train = train[[target_name]]
        X_eval = eval[self.all_features]
        y_eval = eval[[target_name]]

        self.model.fit(
            X_train,
            y_train,
            eval_set=(X_eval, y_eval),
            cat_features=self.cat_feature_indices,
            **fit_params
        )

    def predict(self, test_data):
        X_test = test_data[self.all_features]
        return self.model.predict(X_test)


Feature = namedtuple("feature", "name categorical")

features = [
    Feature("current_runs", False),
    Feature("naive_projection", False),
    Feature("current_wickets", False),
    Feature("overall_ball_n", False),
    Feature("batter_current_runs", False),
    Feature("mean_encoded_batting_team", False),
    # Feature("ten_wicket_prediction", False),
    # Feature("lose_all_wickets", True),
    # Feature("batting_team", True),
    # Feature("batter_won_toss", True),
]

catboost_model = CatboostModel(
    "catboost",
    features,
    {
        "learning_rate": 0.2,
        "iterations": 3000,
        "eval_metric": "RMSE",
        "l2_leaf_reg": 2,
        "max_depth": 3,
    },
)


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

    # features = [Feature("current_runs", False), Feature("batting_team", True)]
    # cat = CatboostModel("cat_model", features, {})

    gkf = GroupKFold(2)

    for train_index, test_index in gkf.split(df, groups=df["match_id"]):

        catboost_model.fit(
            df,
            train_index,
            test_index,
            "first_innings_total_runs",
            {"early_stopping_rounds": 50},
        )

        perm = permutation_importance(
            catboost_model.model,
            df.iloc[test_index][catboost_model.all_features].values,
            df.iloc[test_index][["first_innings_total_runs"]].values,
            scoring="neg_root_mean_squared_error",
        )

        perm_sorted_idx = perm.importances_mean.argsort()
        fig = plt.figure()
        ax = fig.gca()
        y_indices = np.arange(0, len(perm.importances_mean)) + 0.5
        ax.barh(
            y_indices,
            perm.importances_mean[perm_sorted_idx],
            xerr=perm.importances_std[perm_sorted_idx],
        )
        ax.set_yticks(y_indices)
        ax.set_yticklabels(np.array(catboost_model.all_features)[perm_sorted_idx])
        ax.set_xlabel("diff root_mean_squared_error")
        fig.tight_layout()
        fig.savefig("importances.png")

    print("hello")
