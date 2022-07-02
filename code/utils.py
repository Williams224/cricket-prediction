from tempfile import NamedTemporaryFile
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
from sklearn.inspection import permutation_importance
import uuid
import json


def determine_bowl_team(row):
    if row["batting_team"] == row["team_one"]:
        return row["team_two"]
    return row["team_one"]


def rmse_at_balls(test_df, balls, model_name, target_name):
    ball_df = test_df[test_df.overall_ball_n == balls]
    return np.sqrt(
        mean_squared_error(ball_df[target_name], ball_df[f"{model_name}_predictions"])
    )


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


def create_train_test_sets(
    df,
    features_user,
    one_hot_columns,
    target,
    splitter,
    min_bat_team_balls_faced,
    rain_delay_balls_bowled=300,
    total_wickets_lost=10,
):
    df = df[
        ~(
            (df.total_balls_bowled < rain_delay_balls_bowled)
            & (df.total_wickets_lost < total_wickets_lost)
        )
    ]  # remove rain delays

    df = df[df.batting_team_balls_faced > min_bat_team_balls_faced]

    df = df[~df.city.isna()]

    df = pd.get_dummies(df, columns=one_hot_columns, prefix_sep="_-_")

    all_ohc_columns = []
    for ohc in one_hot_columns:
        r = re.compile(f"{ohc}_-_*")
        all_ohc_columns += list(filter(r.match, df.columns))

    features_all = features_user + all_ohc_columns

    split = splitter.split(df, groups=df["match_id"])
    train_inds, test_inds = next(split)
    train = df.iloc[train_inds]
    test = df.iloc[test_inds]
    X_train = train[features_all]
    X_test = test[features_all]
    y_train = train[target]
    y_test = test[target]

    return X_train, X_test, y_train, y_test


def plot_scatter(predicted, actual, label, x_axis="actual", y_axis="predicted"):
    fig = plt.figure()
    ax = fig.gca()
    ax.cla()
    ax.scatter(actual, predicted)
    ax.set_title(label)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    return fig


def scatter_at_balls(df, balls, label, model_name, target_name, **kwargs):
    ball_df = df[df.overall_ball_n == balls]
    scat_fig = plot_scatter(
        ball_df[f"{model_name}_predictions"], ball_df[target_name], label, **kwargs
    )
    scat_fig_fp = f"/tmp/{model_name}_scat_at_{balls}_balls.png"
    scat_fig.savefig(scat_fig_fp)
    return scat_fig_fp


def evaluate_reg(
    data,
    model_name,
    target_name,
    key_balls=[30, 90, 120, 150, 180, 240, 270],
):
    plots = {}
    metrics = {}

    preds = data[f"{model_name}_predictions"]
    y_test = data[target_name]

    scatter_plot = plot_scatter(preds, y_test, f"{model_name}_all balls")
    file = f"/tmp/scatter_plot_{model_name}_all_balls.png"
    scatter_plot.savefig(file)
    plots[f"scatter_plot_{model_name}_all_balls"] = file

    residuals = preds - y_test

    metrics[f"residuals_mean_{model_name}"] = np.mean(residuals)
    metrics[f"residuals_median_{model_name}"] = np.median(residuals)

    resid_fig = plot_residuals(residuals)
    resid_fp = f"/tmp/residuals_hist_{model_name}.png"
    resid_fig.savefig(resid_fp)

    plots[f"residuals_{model_name}"] = resid_fp

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    metrics[f"rmse_{model_name}"] = rmse

    for kb in key_balls:
        rmse_kb = rmse_at_balls(data, kb, model_name, target_name)
        metrics[f"rmse_at_{kb}_{model_name}"] = rmse_kb
        plots[f"{model_name}_scatter_at_{kb}_balls"] = scatter_at_balls(
            data, kb, f"{model_name}_scatter_at_{kb}_balls", model_name, target_name
        )

    rmse_x = []
    rmse_y = []
    for i in range(12, 270):
        rmse_x.append(i)
        rmse_y.append(rmse_at_balls(data, i, model_name, target_name))

    rmse_plot = plot_rmses(rmse_x, rmse_y)
    rmse_plot_fp = f"/tmp/rmse_plot_{model_name}.png"
    rmse_plot.savefig(rmse_plot_fp)

    plots[f"rmse_plot_{model_name}"] = rmse_plot_fp

    return plots, metrics


def get_feature_importances(model, df_test, target_name):
    perm = permutation_importance(
        model,
        df_test[model.all_features],
        df_test[target_name],
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    perm_sorted_idx = perm.importances_mean.argsort()
    fig = plt.figure()
    ax = fig.gca()
    ax.cla()
    y_indices = np.arange(0, len(perm.importances_mean)) + 0.5
    ax.barh(
        y_indices,
        perm.importances_mean[perm_sorted_idx],
        xerr=perm.importances_std[perm_sorted_idx],
    )
    ax.set_yticks(y_indices)
    ax.set_yticklabels(np.array(model.all_features)[perm_sorted_idx])
    ax.set_xlabel("diff root_mean_squared_error")
    fig.tight_layout()
    uuid_fi = str(uuid.uuid4())
    fi_plot_fp = f"/tmp/fi_plot_{model.name}_{uuid_fi}.png"
    fig.savefig(fi_plot_fp)
    fi_dict = {}
    for feature_name, importance_mean, importance_std in zip(
        model.all_features, perm.importances_mean, perm.importances_std
    ):
        fi_dict[feature_name] = [importance_mean, importance_std]

    fi_json_file = f"/tmp/fi_json_{model.name}_{uuid_fi}.png"
    with open(fi_json_file, "w") as f:
        json.dump(fi_dict, f)

    return fi_plot_fp, fi_json_file
