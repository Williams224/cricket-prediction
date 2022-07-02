from base_model import BaseModel
from collections import namedtuple

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder

import pandas as pd


class AdaBoostModel(BaseModel):
    def __init__(self, name, features, model_params):
        super().__init__(name, features)
        self.model = AdaBoostRegressor(
            DecisionTreeRegressor(random_state=69), **model_params
        )
        self.ohc = OneHotEncoder(sparse=False)

    def fit(self, data, train_indices, target_name, fit_params):
        if len(self.cat_features) > 0:
            self.ohc.fit(data[self.cat_features])
            transformed = self.ohc.transform(data[self.cat_features])
            # Create a Pandas DataFrame of the hot encoded column
            ohe_df = pd.DataFrame(transformed, columns=self.ohc.get_feature_names_out())
            # concat with original data
            preprocessed_data = pd.concat([data, ohe_df], axis=1)
            self.preprocessed_feature_columns = list(
                self.ohc.get_feature_names_out()
            ) + list(self.continuous_features)

        else:
            preprocessed_data = data
            self.preprocessed_feature_columns = self.continuous_features

        train = preprocessed_data.iloc[train_indices]
        X_train = train[self.preprocessed_feature_columns]
        y_train = train[[target_name]]

        self.model.fit(
            X_train,
            y_train,
            **fit_params,
        )

    def predict(self, test_data):
        if len(self.cat_features) > 0:
            transformed_cat_data = self.ohc.transform(
                test_data[self.cat_features],
            )
            ohe_df = pd.DataFrame(
                transformed_cat_data, columns=self.ohc.get_feature_names_out()
            )
            preprocessed_data = pd.concat(
                [test_data.reset_index(), ohe_df.reset_index()], axis=1
            )
            X_test = preprocessed_data[self.preprocessed_feature_columns]
        else:
            X_test = test_data[self.preprocessed_feature_columns]

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
    Feature("lose_all_wickets", True),
    Feature("batter_won_toss", True),
]

adaboost_model = AdaBoostModel(
    "adaboost",
    features,
    {"n_estimators": 300, "loss": "exponential", "random_state": 69},
)
