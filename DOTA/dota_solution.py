'''
Solution for "DOTA 2 win prediction" competition.
Makes submission file for result on public LB 0.85360
'''

import os
import pandas as pd
import numpy as np
import time

import ujson as json
import lightgbm as lgb

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import train_test_split, ShuffleSplit, KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm_notebook
from contextlib import contextmanager



PATH_TO_DATA = "../input/"  # Path to competition data

# report running times
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")


print("Reading provided features tables")

df_train_features = pd.read_csv(os.path.join(PATH_TO_DATA, "train_features.csv"), index_col="match_id_hash")
df_train_targets = pd.read_csv(os.path.join(PATH_TO_DATA, "train_targets.csv"), index_col="match_id_hash")
df_test_features = pd.read_csv(os.path.join(PATH_TO_DATA, "test_features.csv"), index_col="match_id_hash")

# reads JSON files by line
def read_matches(matches_file):
    MATCHES_COUNT = {
        "test_matches.jsonl": 10000,
        "train_matches.jsonl": 39675,
    }
    _, filename = os.path.split(matches_file)
    total_matches = MATCHES_COUNT.get(filename)

    with open(matches_file) as fin:
        for line in tqdm_notebook(fin, total=total_matches):
            yield json.loads(line)


# feature engineering - extracting more features from JSOn files
def add_new_features(df_features, matches_file):
    # Process raw data and add new features
    for match in read_matches(matches_file):
        match_id_hash = match["match_id_hash"]

        # Objectives features
        radiant_tower_kills = 0
        dire_tower_kills = 0

        radiant_tower_denies = 0
        dire_tower_denies = 0

        radiant_barrack_kills = 0
        dire_barrack_kills = 0

        radiant_aegis = 0
        dire_aegis = 0

        radiant_aegis_stolen = 0
        dire_aegis_stolen = 0

        radiant_aegis_denied = 0
        dire_aegis_denied = 0

        radiant_first_blood = 0
        dire_first_blood = 0

        radiant_roshan_kill = 0
        dire_roshan_kill = 0

        for objective in match["objectives"]:
            if objective["type"] == "CHAT_MESSAGE_TOWER_KILL":
                if objective["team"] == 2:
                    radiant_tower_kills += 1
                if objective["team"] == 3:
                    dire_tower_kills += 1

            if objective["type"] == "CHAT_MESSAGE_TOWER_DENY":
                if objective["player_slot"] <= 32:
                    radiant_tower_denies += 1
                if objective["player_slot"] > 32:
                    dire_tower_denies += 1

            if objective["type"] == "CHAT_MESSAGE_BARRACKS_KILL":
                if int(objective["key"]) <= 32:
                    radiant_barrack_kills += 1
                if int(objective["key"]) > 32:
                    dire_barrack_kills += 1

            if objective["type"] == "CHAT_MESSAGE_AEGIS":
                if objective["player_slot"] <= 32:
                    radiant_aegis += 1
                if objective["player_slot"] > 32:
                    dire_aegis += 1

            if objective["type"] == "CHAT_MESSAGE_AEGIS_STOLEN":
                if objective["player_slot"] <= 32:
                    radiant_aegis_stolen += 1
                if objective["player_slot"] > 32:
                    dire_aegis_stolen += 1

            if objective["type"] == "CHAT_MESSAGE_DENIED_AEGIS":
                if objective["player_slot"] <= 32:
                    radiant_aegis_denied += 1
                if objective["player_slot"] > 32:
                    dire_aegis_denied += 1

            if objective["type"] == "CHAT_MESSAGE_FIRSTBLOOD":
                if objective["player_slot"] <= 32:
                    radiant_first_blood += 1
                if objective["player_slot"] > 32:
                    dire_first_blood += 1

            if objective["type"] == "CHAT_MESSAGE_ROSHAN_KILL":
                if objective["team"] == 2:
                    radiant_roshan_kill += 1
                if objective["team"] == 3:
                    dire_roshan_kill += 1

        df_features.loc[match_id_hash, "radiant_tower_kills"] = radiant_tower_kills
        df_features.loc[match_id_hash, "dire_tower_kills"] = dire_tower_kills
        df_features.loc[match_id_hash, "diff_tower_kills"] = radiant_tower_kills - dire_tower_kills

        df_features.loc[match_id_hash, "radiant_tower_denies"] = radiant_tower_denies
        df_features.loc[match_id_hash, "dire_tower_denies"] = dire_tower_denies
        df_features.loc[match_id_hash, "diff_tower_denies"] = radiant_tower_denies - dire_tower_denies

        df_features.loc[match_id_hash, "radiant_barrack_kills"] = radiant_barrack_kills
        df_features.loc[match_id_hash, "dire_barrack_kills"] = dire_barrack_kills
        df_features.loc[match_id_hash, "diff_barrack_kills"] = radiant_barrack_kills - dire_barrack_kills

        df_features.loc[match_id_hash, "radiant_aegis"] = radiant_aegis
        df_features.loc[match_id_hash, "dire_aegis"] = dire_aegis
        df_features.loc[match_id_hash, "diff_aegis"] = radiant_aegis - dire_aegis

        df_features.loc[match_id_hash, "radiant_aegis_stolen"] = radiant_aegis_stolen
        df_features.loc[match_id_hash, "dire_aegis_stolen"] = dire_aegis_stolen
        df_features.loc[match_id_hash, "diff_aegis_stolen"] = radiant_aegis_stolen - dire_aegis_stolen

        df_features.loc[match_id_hash, "radiant_aegis_denied"] = radiant_aegis_denied
        df_features.loc[match_id_hash, "dire_aegis_denied"] = dire_aegis_denied
        df_features.loc[match_id_hash, "diff_aegis_denied"] = radiant_aegis_denied - dire_aegis_denied

        df_features.loc[match_id_hash, "radiant_first_blood"] = radiant_first_blood
        df_features.loc[match_id_hash, "dire_first_blood"] = dire_first_blood
        df_features.loc[match_id_hash, "diff_first_blood"] = radiant_first_blood - dire_first_blood

        df_features.loc[match_id_hash, "radiant_roshan_kill"] = radiant_roshan_kill
        df_features.loc[match_id_hash, "dire_roshan_kill"] = dire_roshan_kill
        df_features.loc[match_id_hash, "diff_roshan_kill"] = radiant_roshan_kill - dire_roshan_kill

        # Player extended features
        for slot, player in enumerate(match["players"]):
            if slot < 5:
                player_name = "r%d" % (slot + 1)
            else:
                player_name = "d%d" % (slot - 4)

            df_features.loc[match_id_hash, "%s_%s" % (player_name, "account_id_hash")] = player["account_id_hash"]

            # Gold time series processing
            # Uses beta from linear regression to see how quickly player is gaining gold
            reg = LinearRegression()
            beta = 0
            if len(player["gold_t"]) > 0:
                gld = np.array(player["gold_t"]).reshape(-1, 1)
                xt = np.arange(len(player["gold_t"])).reshape(-1, 1)
                reg.fit(xt, gld)
                beta = reg.coef_[0, 0]
            df_features.loc[match_id_hash, "%s_%s" % (player_name, "gold_t_beta")] = beta

            # More player features
            df_features.loc[match_id_hash, "%s_%s" % (player_name, "ability_upgrades_len")] = len(player["ability_upgrades"])
            df_features.loc[match_id_hash, "%s_%s" % (player_name, "max_hero_hit")] = player["max_hero_hit"]["value"]
            df_features.loc[match_id_hash, "%s_%s" % (player_name, "purchase_log_len")] = len(player["purchase_log"])
            df_features.loc[match_id_hash, "%s_%s" % (player_name, "kills_log_len")] = len(player["kills_log"])
            df_features.loc[match_id_hash, "%s_%s" % (player_name, "buyback_log_len")] = len(player["buyback_log"])
            df_features.loc[match_id_hash, "%s_%s" % (player_name, "runes_log_len")] = len(player["runes_log"])
            df_features.loc[match_id_hash, "%s_%s" % (player_name, "actions_sum")] = np.sum(list(player["actions"].values()))
            df_features.loc[match_id_hash, "%s_%s" % (player_name, "pings_mean")] = np.mean(list(player["pings"].values())) if len(player["pings"].values()) > 0 else 0
            df_features.loc[match_id_hash, "%s_%s" % (player_name, "killed_sum")] = np.sum(list(player["killed"].values()))
            df_features.loc[match_id_hash, "%s_%s" % (player_name, "item_uses_sum")] = np.sum(list(player["item_uses"].values()))
            df_features.loc[match_id_hash, "%s_%s" % (player_name, "ability_uses_sum")] = np.sum(list(player["ability_uses"].values()))
            df_features.loc[match_id_hash, "%s_%s" % (player_name, "hero_hits_sum")] = np.sum(list(player["hero_hits"].values()))
            df_features.loc[match_id_hash, "%s_%s" % (player_name, "damage_sum")] = np.sum(list(player["damage"].values()))
            df_features.loc[match_id_hash, "%s_%s" % (player_name, "damage_mean")] = np.mean(list(player["damage"].values())) if len(player["damage"].values()) > 0 else 0
            df_features.loc[match_id_hash, "%s_%s" % (player_name, "damage_taken_sum")] = np.sum(list(player["damage_taken"].values()))
            df_features.loc[match_id_hash, "%s_%s" % (player_name, "damage_taken_mean")] = np.mean(list(player["damage_taken"].values())) if len(player["damage_taken"].values()) > 0 else 0
            df_features.loc[match_id_hash, "%s_%s" % (player_name, "hero_inventory_len")] = len(player["hero_inventory"])
            df_features.loc[match_id_hash, "%s_%s" % (player_name, "nearby_creep_death_count")] = player["nearby_creep_death_count"]


# Makes one-hot encoding data for hero ids
def make_hero_cvec():
    r_h_ids = ["r1_hero_id", "r2_hero_id", "r3_hero_id", "r4_hero_id", "r5_hero_id"]
    d_h_ids = ["d1_hero_id", "d2_hero_id", "d3_hero_id", "d4_hero_id", "d5_hero_id"]
    r_h_ids_txt = df_train_features[r_h_ids].apply(lambda x: " ".join(x.astype(str)), axis=1)
    d_h_ids_txt = df_train_features[d_h_ids].apply(lambda x: " ".join(x.astype(str)), axis=1)
    r_h_ids_txt_test = df_test_features[r_h_ids].apply(lambda x: " ".join(x.astype(str)), axis=1)
    d_h_ids_txt_test = df_test_features[d_h_ids].apply(lambda x: " ".join(x.astype(str)), axis=1)

    cvec = CountVectorizer()
    r_h_ids_cvec = cvec.fit_transform(r_h_ids_txt)
    r_h_ids_cvec_test = cvec.transform(r_h_ids_txt_test)
    d_h_ids_cvec = cvec.fit_transform(d_h_ids_txt)
    d_h_ids_cvec_test = cvec.transform(d_h_ids_txt_test)

    return r_h_ids_cvec, r_h_ids_cvec_test, d_h_ids_cvec, d_h_ids_cvec_test


# Scales and returns gold feature
def scale_gold():
    features_gold = ["r1_gold_t_beta", "r2_gold_t_beta", "r3_gold_t_beta", "r4_gold_t_beta", "r5_gold_t_beta",
                     "d1_gold_t_beta", "d2_gold_t_beta", "d3_gold_t_beta", "d4_gold_t_beta", "d5_gold_t_beta"]
    X_gold = df_train_features_extended[features_gold].values
    X_gold_test = df_test_features_extended[features_gold].values
    sca = StandardScaler()
    X_gold = sca.fit_transform(X_gold)
    X_gold_test = sca.transform(X_gold_test)
    return X_gold, X_gold_test


# Aggregates most features by teams. Calculates totals, means and standard deviations
# Found in kernel by Andrew Lukyanenko.
def make_aggregates():
    features_agg = []

    for c in agg_cols:
        r_columns = [f"r{i}_{c}" for i in range(1, 6)]
        d_columns = [f"d{i}_{c}" for i in range(1, 6)]

        features_agg += ["r_total_" + c, "d_total_" + c, "total_" + c + "_ratio"]
        df_train_features_extended["r_total_" + c] = df_train_features_extended[r_columns].sum(1)
        df_train_features_extended["d_total_" + c] = df_train_features_extended[d_columns].sum(1)
        df_train_features_extended["total_" + c + "_ratio"] = df_train_features_extended["r_total_" + c] / (df_train_features_extended["d_total_" + c] + 0.1)
        df_test_features_extended["r_total_" + c] = df_test_features_extended[r_columns].sum(1)
        df_test_features_extended["d_total_" + c] = df_test_features_extended[d_columns].sum(1)
        df_test_features_extended["total_" + c + "_ratio"] = df_test_features_extended["r_total_" + c] / (df_test_features_extended["d_total_" + c] + 0.1)

        features_agg += ["r_std_" + c, "d_std_" + c, "std_" + c + "_ratio"]
        df_train_features_extended["r_std_" + c] = df_train_features_extended[r_columns].std(1)
        df_train_features_extended["d_std_" + c] = df_train_features_extended[d_columns].std(1)
        df_train_features_extended["std_" + c + "_ratio"] = df_train_features_extended["r_std_" + c] / (df_train_features_extended["d_std_" + c] + 0.1)
        df_test_features_extended["r_std_" + c] = df_test_features_extended[r_columns].std(1)
        df_test_features_extended["d_std_" + c] = df_test_features_extended[d_columns].std(1)
        df_test_features_extended["std_" + c + "_ratio"] = df_test_features_extended["r_std_" + c] / (df_test_features_extended["d_std_" + c] + 0.1)

        features_agg += ["r_mean_" + c, "d_mean_" + c, "mean_" + c + "_ratio"]
        df_train_features_extended["r_mean_" + c] = df_train_features_extended[r_columns].mean(1)
        df_train_features_extended["d_mean_" + c] = df_train_features_extended[d_columns].mean(1)
        df_train_features_extended["mean_" + c + "_ratio"] = df_train_features_extended["r_mean_" + c] / (df_train_features_extended["d_mean_" + c] + 0.1)
        df_test_features_extended["r_mean_" + c] = df_test_features_extended[r_columns].mean(1)
        df_test_features_extended["d_mean_" + c] = df_test_features_extended[d_columns].mean(1)
        df_test_features_extended["mean_" + c + "_ratio"] = df_test_features_extended["r_mean_" + c] / (df_test_features_extended["d_mean_" + c] + 0.1)

    return features_agg

# Trains LGB model and returns prerdictions
def train_model(X, X_test, y, params, folds):
    """
    Cross-validation LGBM

    cla_lgb = lgb.LGBMClassifier(objective="binary", metric="auc", n_jobs=6, random_state=17)

    # Parameters tested during cross-validation
    lgb_param_grid = {"learning_rate": [0.01], #[0.001, 0.01, 0.1, 0.75, 1.5, 5],
                          "num_boost_round": [4500], #[100, 200, 500, 1000, 3000],
                          "max_depth": [4], #[4, 8, 16, 32, 64, None],
                          "num_leaves": [31], #[31, 50, 100, 200],
                          "min_data_in_leaf": [50], #[2, 10, 50, 100],
                          "unbalance": [True], #[True, False],
                          "feature_fraction": [0.1], #[0.05, 0.1, 0.3, 0.5, 0.7, 0.8, 1.0],
                          "lambda_l1": [0.9], #[0.0, 0.1, 0.5, 0.9],
                          "lambda_l2": [0.9], #[0.0, 0.1, 0.5, 0.9],
                          "max_bin": [255]}

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=17)
    gs = GridSearchCV(estimator=cla_lgb, param_grid=lgb_param_grid, scoring="roc_auc", cv=skf, verbose=3)
    gs.fit(X_cvec_lgb, y)

    print(gs.best_params_)
    """


    prediction = np.zeros(X_test.shape[0])
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
        X_train, X_valid = X.toarray()[train_index], X.toarray()[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid)

        model = lgb.train(params,
                          train_data,
                          num_boost_round=20000,
                          valid_sets=[train_data, valid_data],
                          verbose_eval=-1,
                          early_stopping_rounds=200)

        y_pred = model.predict(X_test, num_iteration=model.best_iteration)

        prediction += y_pred

    prediction /= 5

    return prediction

# Reading extended features
with timer("Reading new features"):
    print("Reading new features")

    df_train_features_extended = df_train_features.copy()

    add_new_features(df_train_features_extended, os.path.join(PATH_TO_DATA, "train_matches.jsonl"))

    df_test_features_extended = df_test_features.copy()

    add_new_features(df_test_features_extended, os.path.join(PATH_TO_DATA, "test_matches.jsonl"))

# Getting hero ids as one-hot encoded data
with timer("CountVectorizing hero ids"):
    print("CountVectorizing hero ids")
    r_h_ids_cvec, r_h_ids_cvec_test, d_h_ids_cvec, d_h_ids_cvec_test = make_hero_cvec()

# Getting gold features
with timer("Making gold features"):
    X_gold, X_gold_test = scale_gold()

# List of objectives features column names
features_extended_obj = ["radiant_tower_kills", "dire_tower_kills", "diff_tower_kills",
                         "radiant_tower_denies", "dire_tower_denies", "diff_tower_denies",
                         "radiant_barrack_kills", "dire_barrack_kills", "diff_barrack_kills",
                         "radiant_aegis", "dire_aegis", "diff_aegis",
                         "radiant_aegis_stolen", "dire_aegis_stolen", "diff_aegis_stolen",
                         "radiant_aegis_denied", "dire_aegis_denied", "diff_aegis_denied",
                         "radiant_first_blood", "dire_first_blood", "diff_first_blood",
                         "radiant_roshan_kill", "dire_roshan_kill", "diff_roshan_kill"]

# Extended features per player which increased rocauc value
features_extended_maxhehil = ["r%s_%s" % (i, "max_hero_hit") for i in range(1,6)] + \
                        ["d%s_%s" % (i, "max_hero_hit") for i in range(1,6)]

features_extended_kllogl = ["r%s_%s" % (i, "kills_log_len") for i in range(1,6)] + \
                        ["d%s_%s" % (i, "kills_log_len") for i in range(1,6)]

features_extended_runll = ["r%s_%s" % (i, "runes_log_len") for i in range(1,6)] + \
                        ["d%s_%s" % (i, "runes_log_len") for i in range(1,6)]

features_extended_abuss = ["r%s_%s" % (i, "ability_uses_sum") for i in range(1,6)] + \
                        ["d%s_%s" % (i, "ability_uses_sum") for i in range(1,6)]

# Names of columns to aggregate by team
agg_cols = ["kills", "deaths", "assists", "denies", "gold", "lh", "xp", "health", "max_health", "max_mana", "level",
          "x", "y", "stuns", "creeps_stacked", "camps_stacked", "rune_pickups",
          "firstblood_claimed", "teamfight_participation", "towers_killed", "roshans_killed", "obs_placed",
          "sen_placed",
           "ability_upgrades_len", "max_hero_hit", "purchase_log_len",
            "kills_log_len", "buyback_log_len", "runes_log_len", "actions_sum", "pings_mean", "killed_sum",
            "item_uses_sum", "ability_uses_sum", "hero_hits_sum", "damage_sum", "damage_mean", "damage_taken_sum",
            "damage_taken_mean", "hero_inventory_len", "nearby_creep_death_count",
            "gold_t_beta"]

# Aggregates features
with timer("Aggregating features"):
    print("Aggregating features")
    features_agg = make_aggregates()

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
sca = StandardScaler()
ohe = OneHotEncoder(categories="auto")

# Join features for train and test
X_cvec_lgb = hstack([r_h_ids_cvec, d_h_ids_cvec,
                 X_gold, df_train_features_extended[features_extended_obj].values,
                 df_train_features_extended[features_extended_abuss].values,
                 df_train_features_extended[features_extended_kllogl].values,
                 df_train_features_extended[features_extended_maxhehil].values,
                 df_train_features_extended[features_extended_runll].values,
                 sca.fit_transform(df_train_features_extended[features_agg].values),
                 ohe.fit_transform(df_train_features[["lobby_type", "game_mode"]].values)])

X_cvec_test_lgb = hstack([r_h_ids_cvec_test, d_h_ids_cvec_test,
                 X_gold_test, df_test_features_extended[features_extended_obj].values,
                 df_test_features_extended[features_extended_abuss].values,
                 df_test_features_extended[features_extended_kllogl].values,
                 df_test_features_extended[features_extended_maxhehil].values,
                 df_test_features_extended[features_extended_runll].values,
                 sca.transform(df_test_features_extended[features_agg].values),
                 ohe.transform(df_test_features_extended[["lobby_type", "game_mode"]].values)])

y = df_train_targets["radiant_win"].values

# Best LBG parameters found in cross-validation
params = {"boost": "gbdt",
          "feature_fraction": 0.1,
          "learning_rate": 0.01,
          "max_depth": 4,
          "lambda_l1": 0.9,
          "lambda_l2": 0.9,
          "metric": "auc",
          "min_data_in_leaf": 50,
          "num_leaves": 31,
          "num_threads": -1,
          "verbosity": -1,
          "objective": "binary"
         }

# Applying LGBM
with timer("LightGBM: train and predict"):
    pred = train_model(X_cvec_lgb, X_cvec_test_lgb, y, params=params, folds=folds)

# Writing submission
with timer("Prepare submission"):
    df_submission = pd.DataFrame(
        {"radiant_win_prob": pred},
        index=df_test_features.index,
    )
    df_submission.to_csv(os.path.join(PATH_TO_DATA, "submission_dota.csv"))