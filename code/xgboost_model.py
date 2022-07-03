from cgi import test
from base_model import BaseModel
from collections import namedtuple
import xgboost as xgb
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder


class XGBoostModel(BaseModel):
    def __init__(self, name, features, model_params):
        super().__init__(name, features)
        self.model = xgb.XGBRegressor(**model_params)
        self.ohc = OneHotEncoder(sparse=False)

    def fit(self, data, train_indices, eval_indices, target_name, fit_params):
        self.ohc.fit(data[self.cat_features])
        transformed = self.ohc.transform(data[self.cat_features])
        # Create a Pandas DataFrame of the hot encoded column
        ohe_df = pd.DataFrame(transformed, columns=self.ohc.get_feature_names_out())
        # concat with original data
        preprocessed_data = pd.concat([data, ohe_df], axis=1)
        self.preprocessed_feature_columns = list(
            self.ohc.get_feature_names_out()
        ) + list(self.continuous_features)

        train = preprocessed_data.iloc[train_indices]
        eval = preprocessed_data.iloc[eval_indices]
        X_train = train[self.preprocessed_feature_columns]
        y_train = train[[target_name]]
        X_eval = eval[self.preprocessed_feature_columns]
        y_eval = eval[[target_name]]

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_eval, y_eval)],
            **fit_params,
        )

    def predict(self, test_data):
        transformed_cat_data = self.ohc.transform(
            test_data[self.cat_features],
        )
        ohe_df = pd.DataFrame(
            transformed_cat_data, columns=self.ohc.get_feature_names_out()
        )
        preprocessed_data = pd.concat(
            [test_data.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1
        )
        X_test = preprocessed_data[self.preprocessed_feature_columns]
        return self.model.predict(X_test)


Feature = namedtuple("feature", "name categorical")

features = [
    Feature("current_runs", False),
    Feature("naive_projection", False),
    Feature("current_wickets", False),
    Feature("overall_ball_n", False),
    Feature("batter_current_runs", False),
    Feature("mean_encoded_batting_team", False),
    Feature("ten_wicket_prediction", False),
    # Feature("lose_all_wickets", True),
    # Feature("batting_team", True),
    # Feature("batter_won_toss", True),
]


xgboost_model = XGBoostModel(
    "xgboost",
    features,
    {
        "objective": "reg:squarederror",
        "colsample_bytree": 0.3,
        "learning_rate": 0.2,
        "n_estimators": 3000,
        "max_depth": 4,
        "early_stopping_rounds": 50,
    },
)

if __name__ == "__main__":
    df = pd.read_parquet(
        "/Users/TimothyW/Fun/cricket_prediction/data/processed_first_innings/first_innings_processed.parquet"
    ).reset_index()

    xgboost_model.fit(
        df, np.arange(0, 2000), np.arange(2000, 3000), "first_innings_total_runs", {}
    )

    xgboost_model.predict(df.iloc[np.arange(2000, 10000)])

    print("hello")
