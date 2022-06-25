from tempfile import NamedTemporaryFile
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd


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


def scatter_at_balls(df, balls, label, path, **kwargs):
    ball_df = df[df.overall_ball_n == balls]
    scat_fig = plot_scatter(ball_df["preds"], ball_df["actual"], label, **kwargs)
    scat_fig_fp = f"{path}/scat_at_{balls}_balls.png"
    scat_fig.savefig(scat_fig_fp)
    return scat_fig_fp


def evaluate_reg(
    data,
    model_prediction_column,
    target_name,
    key_balls=[30, 90, 120, 150, 180, 240, 270],
):
    plots = {}

    scatter_plot = plot_scatter(
        data[model_prediction_column], data[target_name], "all balls"
    )
    with NamedTemporaryFile() as tf:
        scatter_plot.savefig(tf.name)
        plots["scatter_plot"] = tf.name

    return plots

    #
    #   plots["scatter_plot"] = scatter_plot_fp

    #  residuals = preds - flat_y_test

    #   resid_fig = plot_residuals(residuals)
    #   resid_fp = f"{plots_path}/residuals_hist.png"
    #   resid_fig.savefig(resid_fp)

    #  plots["residuals"] = resid_fp

    #  X_test["preds"] = preds
    #  X_test["actual"] = y_test
    #  X_test["residuals"] = preds - flat_y_test

    #  rmse = np.sqrt(mean_squared_error(y_test, preds))

    #  metrics = {"RMSE": rmse}

    ##  mid_way = X_test[X_test.overall_ball_n == 150]

    #  rmse_halfway = np.sqrt(mean_squared_error(mid_way["actual"], mid_way["preds"]))


#    metrics["rmse_halfway"] = rmse_halfway


##  for kb in key_balls:
#      rmse_kb = rmse_at_balls(X_test, kb)
#      metrics[f"rmse_at_{kb}"] = rmse_kb
#
#       plots[f"scatter_at_{kb}_balls"] = scatter_at_balls(
#          X_test, kb, f"scatter_at_{kb}_balls", plots_path
##      )

#  rmse_x = []
##   rmse_y = []
#   for i in range(12, 270):
#     rmse_x.append(i)
#     rmse_y.append(rmse_at_balls(X_test, i))
#
#  rmse_plot = plot_rmses(rmse_x, rmse_y)
#   rmse_plot_fp = f"{plots_path}/rmse_plot.png"
#  rmse_plot.savefig(rmse_plot_fp)

#  plots["rmse_plot"] = rmse_plot_fp

#  return plots, metrics
