from base_model import BaseModel
from collections import namedtuple
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np


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
    Feature("current_wickets", False),
    Feature("overall_ball_n", False),
    Feature("batter_current_runs", False),
    Feature("batting_team", True),
    Feature("batter_won_toss", True),
]

catboost_model = CatboostModel(
    "catboost_model",
    features,
    {
        "learning_rate": 0.1,
        "iterations": 3000,
        "eval_metric": "RMSE",
        "l2_leaf_reg": 2,
        "max_depth": 3,
    },
)


if __name__ == "__main__":
    df = pd.read_parquet(
        "/Users/TimothyW/Fun/cricket_prediction/data/processed_first_innings/first_innings_processed.parquet"
    )

    features = [Feature("current_runs", False), Feature("batting_team", True)]
    cat = CatboostModel("cat_model", features, {})

    cat.fit(
        df,
        np.arange(0, 2000),
        np.arange(2000, 3000),
        "first_innings_total_runs",
        {"early_stopping_rounds": 50},
    )

    print("hello")
