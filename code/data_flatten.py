import pandas as pd
import json
import numpy as np
import os


# Aim to flatten everything so each delivery is a flat dict
# start with a single delivery

# ultimate aim is df with  wickets | runs | balls left | target_runs


def process_delivery(jsn):
    if "wickets" in jsn.keys():
        wicket_player_out = jsn["wickets"][0]["player_out"]
        wicket_kind = jsn["wickets"][0]["kind"]
        wicket_taken = True
    else:
        wicket_player_out = None
        wicket_kind = None
        wicket_taken = False
    return {
        "batter": jsn["batter"],
        "bowler": jsn["bowler"],
        "non_striker": jsn["non_striker"],
        "batter_runs": jsn["runs"]["batter"],
        "extras_runs": jsn["runs"]["extras"],
        "total_delivery_runs": jsn["runs"]["total"],
        "wicket_player_out": wicket_player_out,
        "wicket_kind": wicket_kind,
        "wicket_taken": wicket_taken,
    }


def process_over(jsn):
    deliv_list = []
    ref_dict = {"over_n": jsn["over"]}
    delivery_list = jsn["deliveries"]
    fraction_ball = 1.0 / len(delivery_list)
    for ball_i in range(0, len(delivery_list)):
        delivery = process_delivery(delivery_list[ball_i])
        delivery["over_ball_n"] = ball_i + 1
        delivery["over_fractional"] = ball_i * fraction_ball
        delivery["overall_ball_fraction"] = jsn["over"] + delivery["over_fractional"]
        delivery_ref = dict(ref_dict, **delivery)
        deliv_list.append(delivery_ref)

    return deliv_list


def process_innings(jsn):
    over_list = []
    for over in jsn["overs"]:
        pr_over = process_over(over)
        over_list += pr_over

    innings_df = pd.DataFrame(over_list)
    innings_df["batting_team"] = jsn["team"]
    innings_df["first_innings_total_runs"] = innings_df["total_delivery_runs"].sum()
    innings_df["current_runs"] = innings_df["total_delivery_runs"].cumsum()
    innings_df["current_wickets"] = innings_df["wicket_taken"].cumsum()
    innings_df["overall_ball_n"] = np.arange(len(innings_df))
    innings_df["total_balls_bowled"] = len(innings_df)
    innings_df["innings_balls_left"] = len(innings_df) - innings_df["overall_ball_n"]
    innings_df["total_wickets_lost"] = innings_df["wicket_taken"].sum()

    return innings_df


def process_match(jsn):
    first_innings_df = process_innings(jsn["innings"][0])
    first_innings_df["data_version"] = jsn["meta"]["data_version"]
    first_innings_df["date_created"] = jsn["meta"]["created"]
    first_innings_df["revision"] = jsn["meta"]["revision"]
    first_innings_df["city"] = jsn["info"].get("city", None)
    first_innings_df["date"] = jsn["info"]["dates"][0]
    first_innings_df["match_type"] = jsn["info"]["match_type"]
    first_innings_df["toss_winner"] = jsn["info"]["toss"]["winner"]
    first_innings_df["team_one"] = jsn["info"]["teams"][0]
    first_innings_df["team_two"] = jsn["info"]["teams"][1]
    i = 0
    for team, players in jsn["info"]["players"].items():
        if i == 0:
            team_label = "team_one"
        elif i == 1:
            team_label = "team_two"
        for p in range(0, len(players)):
            first_innings_df[f"{team_label}_player_{p}"] = players[p]
        i = i + 1

    return first_innings_df


def process_file(f_name):
    with open(f_name, "r") as f:
        d = json.load(f)
    return process_match(d)


if __name__ == "__main__":

    data_path = "/Users/TimothyW/Fun/cricket_prediction/data/odis_male_json/"
    dfs = []
    for f_name in os.listdir(data_path):
        f_name_full = f"{data_path}/{f_name}"
        dfs.append(process_file(f_name_full))

    df_all = pd.concat(dfs)

    df_all.info(verbose=True)

    df_all.to_parquet(
        "/Users/TimothyW/Fun/cricket_prediction/data/processed_first_innings/first_innings_processed.parquet"
    )
