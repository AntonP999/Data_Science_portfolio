import os
import numpy as np
import pandas as pd
import json
from tqdm import tqdm_notebook
import time
from contextlib import contextmanager
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from scipy.sparse import csr_matrix, hstack
import lightgbm as lgb
from html.parser import HTMLParser
import re
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
stopwords = stopwords.words("english")

PATH_TO_DATA = "../input"  # Path to competition data
SEED = 17
TRAIN_LEN = 62313  # just for tqdm to see progress
TEST_LEN = 34645  # just for tqdm to see progress
CONTENT_NGRAMS = (1, 2)  # for tf-idf on content
TITLE_NGRAMS = (1, 3)  # for tf-idf on titles
MAX_FEATURES = 100000  # for tf-idf
LGB_TRAIN_ROUNDS = 119  # num. iteration to train LightGBM
MEAN_TEST_TARGET = 4.33328  # what we got by submitting all zeros
RIDGE_WEIGHT = 0.6  # weight of Ridge predictions in a blend with LightGBM
letters = re.compile(r"\w+", re.UNICODE)
lemmer = WordNetLemmatizer()

# report running times
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")


# The following code will help to throw away all HTML tags from article content/title
class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return "".join(self.fed)


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


# Supplementary function to read a JSON line without crashing on escape characters.
def read_json_line(line=None):
    result = None
    try:
        result = json.loads(line)
    except Exception as e:
        # Find the offending character index:
        idx_to_replace = int(str(e).split(" ")[-1].replace(")",""))
        # Remove the offending character:
        new_line = list(line)
        new_line[idx_to_replace] = " "
        new_line = "".join(new_line)
        return read_json_line(line=new_line)
    return result


# feature engineering - extracting titles from raw JSOn files
def extract_features_and_write(path_to_data, inp_filename, is_train=True):
    features = ["content", "published", "title", "author"]
    prefix = "train" if is_train else "test"
    total = TRAIN_LEN if is_train else TEST_LEN
    feature_files = [open(os.path.join(path_to_data,
                                       "{}_{}.txt".format(prefix, feat)),
                          "w", encoding="utf-8")
                     for feat in features]

    with open(os.path.join(path_to_data, inp_filename),
              encoding="utf-8") as inp_json_file:
        for line in tqdm_notebook(inp_json_file, total=total):
            json_data = read_json_line(line)

            content = json_data["content"].replace("\n", " ").replace("\r", " ")
            content_no_html_tags = strip_tags(content)
            published = json_data["published"]["$date"]
            title = json_data["title"].replace("\n", " ").replace("\r", " ")
            title_no_html_tags = strip_tags(title)
            author = json_data["author"]
            author_str = ",".join([str(v) for v in author.values()])

            feature_files[0].write(content_no_html_tags + "\n")
            feature_files[1].write(published + "\n")
            feature_files[2].write(title_no_html_tags + "\n")
            feature_files[3].write(author_str + "\n")


# Extracts domain name and min_read from meta
def extract_domain_meta(path_to_data, inp_filename, total=None):
    domains = []
    metas = []

    with open(os.path.join(path_to_data, inp_filename),
              encoding="utf-8") as inp_json_file:
        for line in tqdm_notebook(inp_json_file, total=total):
            json_data = read_json_line(line)
            domains.append(json_data["domain"].replace("\n", " ").replace("\r", " "))
            metas.append(int(json_data["meta_tags"]["twitter:data1"].split()[0]))

    return domains, metas


# Lemmatizes text
def lem_post(data):
    text_l = letters.findall(str(data))
    return " ".join([lemmer.lemmatize(i, pos="v") for i in text_l if i not in stopwords])


# Tf-idf content
def vectorize_content_lemmed(ngram_range=CONTENT_NGRAMS, max_features=MAX_FEATURES):
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
    X_train_content_sparse = vectorizer.fit_transform(open(os.path.join(PATH_TO_DATA, "content_lemmed_train.txt"), encoding="utf-8"))
    X_test_content_sparse = vectorizer.transform(open(os.path.join(PATH_TO_DATA, "content_lemmed_test.txt"), encoding="utf-8"))
    return X_train_content_sparse, X_test_content_sparse

# Tf-idf titles
def vectorize_title_lemmed(ngram_range=TITLE_NGRAMS, max_features=MAX_FEATURES):
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
    X_train_title_sparse = vectorizer.fit_transform(open(os.path.join(PATH_TO_DATA, "title_lemmed_train.txt"), encoding="utf-8"))
    X_test_title_sparse = vectorizer.transform(open(os.path.join(PATH_TO_DATA, "title_lemmed_test.txt"), encoding="utf-8"))
    return X_train_title_sparse, X_test_title_sparse


def make_files():
    print("Extracting json's")
    extract_features_and_write(PATH_TO_DATA, 'train.json', is_train=True)
    extract_features_and_write(PATH_TO_DATA, 'test.json', is_train=False)

    print("Lemmatizing content")
    lemmed_files = [open(os.path.join(PATH_TO_DATA, "content_lemmed_train.txt"), "w", encoding="utf-8"),
                    open(os.path.join(PATH_TO_DATA, "content_lemmed_test.txt"), "w", encoding="utf-8")]

    with open(os.path.join(PATH_TO_DATA, "train_content.txt"), encoding="utf-8") as input_file:
        for line in tqdm_notebook(input_file, total=62313):
            line_lemmed = lem_post(line.lower())
            lemmed_files[0].write(line_lemmed + "\n")

    with open(os.path.join(PATH_TO_DATA, "test_content.txt"), encoding="utf-8") as input_file:
        for line in tqdm_notebook(input_file, total=34645):
            line_lemmed = lem_post(line.lower())
            lemmed_files[1].write(line_lemmed + "\n")

    print("Lemmatizing titles")
    lemmed_files = [open(os.path.join(PATH_TO_DATA, "title_lemmed_train.txt"), "w", encoding="utf-8"),
                    open(os.path.join(PATH_TO_DATA, "title_lemmed_test.txt"), "w", encoding="utf-8")]

    with open(os.path.join(PATH_TO_DATA, "train_title.txt"), encoding="utf-8") as input_file:
        for line in input_file:
            line_lemmed = lem_post(line.lower())
            lemmed_files[0].write(line_lemmed + "\n")

    with open(os.path.join(PATH_TO_DATA, "test_title.txt"), encoding="utf-8") as input_file:
        for line in input_file:
            line_lemmed = lem_post(line.lower())
            lemmed_files[1].write(line_lemmed + "\n")

    del lemmed_files

def prepare_train_and_test():
    df_train_author = pd.read_csv(os.path.join(PATH_TO_DATA, "train_author.txt"), header=None,
                                  names=["name", "url", "twitter"])
    df_train_published = pd.read_csv(os.path.join(PATH_TO_DATA, "train_published.txt"), header=None, names=["time"],
                                     parse_dates=["time"])
    df_test_author = pd.read_csv(os.path.join(PATH_TO_DATA, "test_author.txt"), header=None,
                                 names=["name", "url", "twitter"])
    df_test_published = pd.read_csv(os.path.join(PATH_TO_DATA, "test_published.txt"), header=None, names=["time"],
                                    parse_dates=["time"])

    print("Getting domains and meta")
    domains_train, metas_train = extract_domain_meta(PATH_TO_DATA, "train.json", total=62313)
    domains_test, metas_test = extract_domain_meta(PATH_TO_DATA, "test.json", total=34645)

    print("Getting content and title lengths")
    pub_lens_train = []
    with open(os.path.join(PATH_TO_DATA, "train_content.txt"), encoding="utf-8") as input_file:
        for line in input_file:
            pub_lens_train.append(len(line))

    pub_lens_test = []
    with open(os.path.join(PATH_TO_DATA, "test_content.txt"), encoding="utf-8") as input_file:
        for line in input_file:
            pub_lens_test.append(len(line))

    title_lens_train = []
    with open(os.path.join(PATH_TO_DATA, "train_title.txt"), encoding="utf-8") as input_file:
        for line in input_file:
            title_lens_train.append(len(line))

    title_lens_test = []
    with open(os.path.join(PATH_TO_DATA, "test_title.txt"), encoding="utf-8") as input_file:
        for line in input_file:
            title_lens_test.append(len(line))

    pub_lens_train = np.array(pub_lens_train).astype(np.float32)
    pub_lens_test = np.array(pub_lens_test).astype(np.float32)
    title_lens_train = np.array(title_lens_train).astype(np.float32)
    title_lens_test = np.array(title_lens_test).astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(np.c_[pub_lens_train, title_lens_train])
    X_train_len_features_scaled = scaler.transform(np.c_[pub_lens_train, title_lens_train])
    X_test_len_features_scaled = scaler.transform(np.c_[pub_lens_test, title_lens_test])

    print("Getting time features")
    df_train_published["year"] = pd.Index(df_train_published["time"]).year
    df_train_published["quarter"] = pd.Index(df_train_published["time"]).quarter
    df_train_published["month"] = pd.Index(df_train_published["time"]).month
    df_train_published["week"] = pd.Index(df_train_published["time"]).week
    df_train_published["dayofyear"] = pd.Index(df_train_published["time"]).dayofyear
    df_train_published["dayofweek"] = pd.Index(df_train_published["time"]).dayofweek
    df_train_published["dayofmonth"] = pd.Index(df_train_published["time"]).day
    df_train_published["hour"] = pd.Index(df_train_published["time"]).hour
    df_train_published["minute"] = pd.Index(df_train_published["time"]).minute

    df_test_published["year"] = pd.Index(df_test_published["time"]).year
    df_test_published["quarter"] = pd.Index(df_test_published["time"]).quarter
    df_test_published["month"] = pd.Index(df_test_published["time"]).month
    df_test_published["week"] = pd.Index(df_test_published["time"]).week
    df_test_published["dayofyear"] = pd.Index(df_test_published["time"]).dayofyear
    df_test_published["dayofweek"] = pd.Index(df_test_published["time"]).dayofweek
    df_test_published["dayofmonth"] = pd.Index(df_test_published["time"]).day
    df_test_published["hour"] = pd.Index(df_test_published["time"]).hour
    df_test_published["minute"] = pd.Index(df_test_published["time"]).minute

    df_train_published["morning"] = df_train_published["hour"].apply(lambda x: 1 if (x >= 7) & (x <= 11) else 0)
    df_train_published["day"] = df_train_published["hour"].apply(lambda x: 1 if (x >= 12) & (x <= 18) else 0)
    df_train_published["evening"] = df_train_published["hour"].apply(lambda x: 1 if (x >= 19) & (x <= 23) else 0)
    df_train_published["night"] = df_train_published["hour"].apply(lambda x: 1 if (x >= 0) & (x <= 6) else 0)

    df_test_published["morning"] = df_test_published["hour"].apply(lambda x: 1 if (x >= 7) & (x <= 11) else 0)
    df_test_published["day"] = df_test_published["hour"].apply(lambda x: 1 if (x >= 12) & (x <= 18) else 0)
    df_test_published["evening"] = df_test_published["hour"].apply(lambda x: 1 if (x >= 19) & (x <= 23) else 0)
    df_test_published["night"] = df_test_published["hour"].apply(lambda x: 1 if (x >= 0) & (x <= 6) else 0)

    df_train_published["weekend"] = df_train_published["dayofweek"].apply(lambda x: 1 if x >= 5 else 0)
    df_test_published["weekend"] = df_test_published["dayofweek"].apply(lambda x: 1 if x >= 5 else 0)

    time_features_toscale = ["year", "week", "dayofyear", "dayofmonth", "minute"]
    time_features_toohe = ["quarter", "month", "dayofweek", "hour"]
    time_features = ["morning", "day", "evening", "night", "weekend"]

    scaler = StandardScaler()
    scaler.fit(df_train_published[time_features_toscale])
    X_train_time_features_scaled = scaler.transform(df_train_published[time_features_toscale])
    X_test_time_features_scaled = scaler.transform(df_test_published[time_features_toscale])

    ohe = OneHotEncoder()
    X_train_time_features_ohe = ohe.fit_transform(df_train_published[time_features_toohe])
    X_test_time_features_ohe = ohe.transform(df_test_published[time_features_toohe])

    X_train_time_features_sparse = df_train_published[time_features].values
    X_test_time_features_sparse = df_test_published[time_features].values

    X_train_time_features_sparse = hstack([X_train_time_features_sparse, X_train_time_features_ohe,
                                           X_train_time_features_scaled]).tocsr()
    X_test_time_features_sparse = hstack([X_test_time_features_sparse, X_test_time_features_ohe,
                                          X_test_time_features_scaled]).tocsr()

    print("Getting author features")
    df_train_author["name"] = df_train_author["url"].apply(lambda x: x.split("@")[-1])
    df_test_author["name"] = df_test_author["url"].apply(lambda x: x.split("@")[-1])

    idx_split = df_train_author.shape[0]
    df_full_author = pd.concat([df_train_author, df_test_author])

    # Labels names for OHE and gets number of publications
    gr = df_full_author.groupby("name")["name"]
    name_dict = {}
    for k, v in enumerate(gr.size().index):
        name_dict[v] = k

    num_pub_dict = gr.size().to_dict()

    df_train_author["name_num"] = df_train_author["name"].map(name_dict)
    df_train_author["pub_num"] = df_train_author["name"].map(num_pub_dict)

    df_test_author["name_num"] = df_test_author["name"].map(name_dict)
    df_test_author["pub_num"] = df_test_author["name"].map(num_pub_dict)

    df_full_author["name_num"] = df_full_author["name"].map(name_dict)
    df_full_author["pub_num"] = df_full_author["name"].map(num_pub_dict)

    ohe = OneHotEncoder()
    X_train_author_sparse = ohe.fit_transform(df_train_author["name_num"].values.reshape(-1, 1))
    X_test_author_sparse = ohe.transform(df_test_author["name_num"].values.reshape(-1, 1))

    df_train_author["has_twitter"] = df_train_author["twitter"].apply(lambda x: 0 if x == "None" else 1)
    df_test_author["has_twitter"] = df_test_author["twitter"].apply(lambda x: 0 if x == "None" else 1)

    df_train_author["domain"] = domains_train
    df_test_author["domain"] = domains_test
    df_train_author["min_read"] = np.log(metas_train)
    df_test_author["min_read"] = np.log(metas_test)

    dom_dict = pd.concat([df_train_author, df_test_author]).groupby("domain").size().to_dict()
    df_train_author["domain_lbl"] = df_train_author["domain"].map(dom_dict)
    df_test_author["domain_lbl"] = df_test_author["domain"].map(dom_dict)

    scaler = StandardScaler()
    scaler.fit(df_train_author[["domain_lbl", "min_read"]].values)
    X_train_domain_meta_scaled = scaler.transform(df_train_author[["domain_lbl", "min_read"]].values)
    X_test_domain_meta_scaled = scaler.transform(df_test_author[["domain_lbl", "min_read"]].values)

    print("Vectorizing content")
    X_train_content_sparse, X_test_content_sparse = vectorize_content_lemmed()
    print("Vectorizing titles")
    X_train_title_sparse, X_test_title_sparse = vectorize_title_lemmed()

    scaler = StandardScaler()
    scaler.fit(df_train_author["pub_num"].values.reshape(-1, 1).astype(np.float32))
    X_train_author_pn_tw = scaler.transform(df_train_author["pub_num"].values.reshape(-1, 1).astype(np.float32))
    X_test_author_pn_tw = scaler.transform(df_test_author["pub_num"].values.reshape(-1, 1).astype(np.float32))

    X_train_author_pn_tw = np.c_[X_train_author_pn_tw, df_train_author["has_twitter"].values]
    X_test_author_pn_tw = np.c_[X_test_author_pn_tw, df_test_author["has_twitter"].values]

    X_train = hstack([X_train_content_sparse, X_train_title_sparse,
                             csr_matrix(X_train_author_sparse), X_train_author_pn_tw[:, 1].reshape(-1, 1),
                             X_train_time_features_sparse, X_train_len_features_scaled,
                             X_train_domain_meta_scaled]).tocsr()
    X_test = hstack([X_test_content_sparse, X_test_title_sparse,
                            csr_matrix(X_test_author_sparse), X_test_author_pn_tw[:, 1].reshape(-1, 1),
                            X_test_time_features_sparse, X_test_len_features_scaled, X_test_domain_meta_scaled]).tocsr()

    train_target = pd.read_csv(os.path.join(PATH_TO_DATA, "train_log1p_recommends.csv"),
                               index_col="id")
    y_train = train_target["log_recommends"].values

    return X_train, y_train, X_test


def ridge_prediction(X_train, y_train, X_test):
    # Cross-validation found alpha=0.001
    # reg = RidgeCV(alphas=(0.001, 0.01, 0.1, 1), scoring="neg_mean_absolute_error", cv=3)
    ridge = Ridge(alpha=0.001, random_state=SEED)
    ridge.fit(X_train, y_train)
    ridge_test_pred = ridge.predict(X_test)
    return ridge_test_pred


def lightgbm_prediction(X_train, y_train, X_test):
    """
    Cross-validation LGBM

    from sklearn.model_selection import GridSearchCV

    lgb_param_grid = {"learning_rate": [0.25], #[0.001, 0.01, 0.1, 0.25, 0.75],
                     "max_depth": [16], #[4, 8, 16, None],
                     "num_leaves": [50]}

    reg_lgb3 = lgb.LGBMRegressor(application="regression", metric="MAE", n_jobs=-1)
    grd_search = GridSearchCV(reg_lgb3, param_grid=lgb_param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=1, verbose=3)
    grd_search.fit(X_train_sparse, y_train)
    print(grd_search.best_params_)

    lgb_x_train_part = lgb.Dataset(X_train_part_sparse.astype(np.float32),
                               label=y_train_part)
    lgb_x_valid = lgb.Dataset(X_valid_sparse.astype(np.float32),
                          label=y_valid)

    param = {"learning_rate": 0.25,
             "max_depth": 16,
             "objective": "mean_absolute_error",
             "metric": "mae"}

    num_round = 200
    bst_lgb = lgb.train(param, lgb_x_train_part, num_round,
                        valid_sets=[lgb_x_valid], early_stopping_rounds=20)

    num_rounds = 119
    """

    lgb_x_train = lgb.Dataset(X_train.astype(np.float32), label=y_train)
    lgb_params = {"learning_rate": 0.25,
                  "max_depth": 16,
                  "seed": SEED,
                  "objective": "mean_absolute_error",
                  "metric": "mae"}

    lgb_model = lgb.train(lgb_params, lgb_x_train, LGB_TRAIN_ROUNDS)
    lgb_test_pred = lgb_model.predict(X_test.astype(np.float32))
    return lgb_test_pred


def form_final_prediction(ridge_pred, lgb_pred, y_train, ridge_weight):
    # blending predictions of Ridge and LightGBM
    mix_pred = ridge_weight * ridge_pred + (1 - ridge_weight) * lgb_pred

    # leaderboard probing
    mix_test_pred_modif = mix_pred + MEAN_TEST_TARGET - y_train.mean()
    return mix_test_pred_modif


with timer("Writing files"):
    make_files()

with timer("Preparing data"):
    X_train, y_train, X_test = prepare_train_and_test()

with timer("Ridge: train and predict"):
    ridge_test_pred = ridge_prediction(X_train, y_train, X_test)

with timer("LightGBM: train and predict"):
    lgb_test_pred = lightgbm_prediction(X_train, y_train, X_test)

with timer("Prepare submission"):
    test_pred = form_final_prediction(ridge_test_pred, lgb_test_pred,
                                      y_train, ridge_weight=RIDGE_WEIGHT)
    submission_df = pd.read_csv(os.path.join(PATH_TO_DATA,
                                             "sample_submission.csv"), index_col="id")
    submission_df["log_recommends"] = test_pred
    submission_df.to_csv(f"submission_medium.csv")