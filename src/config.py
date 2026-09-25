"""Central configuration. Override the data location with the ER_DATA_DIR env var."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_CANDIDATES = [os.path.join(ROOT, "student_resource", "dataset"), r"D:\student_resource\dataset"]
DATA_DIR = os.environ.get("ER_DATA_DIR") or next(
    (d for d in _DATA_CANDIDATES if os.path.isdir(d)), _DATA_CANDIDATES[0])
OUT_DIR = os.environ.get("ER_OUT_DIR", os.path.join(ROOT, "output"))
WORK_DIR = os.environ.get("ER_WORK_DIR", os.path.join(ROOT, "work"))
VALIDATOR = os.path.join(os.path.dirname(DATA_DIR), "utils", "validate_submission.py")

SEED = 42
N_FOLDS = 5
N_JOBS = max(1, (os.cpu_count() or 2) - 1)

# ---- blocking ----
# Restrict candidates to records with the same country label (empty labels are compared with all).
# Flip to False if EDA shows cross-country matches in the ground truth.
BLOCK_BY_COUNTRY = True
K_NAME = 25        # S1 -> S2/S3 top-k on name char TF-IDF
K_COMBO = 25       # S1 -> S2/S3 top-k on name+address char TF-IDF
K_ADDR = 5         # S1 -> S2/S3 top-k on address char TF-IDF
K_REV = 3          # S2/S3 -> S1 top-k (reverse direction), on name and on combo
MIN_SIM = 0.05
# dense retriever (src/dense.py, run on Kaggle): add its top DENSE_K per S1 to the key candidates; cos/rank
# features are looked up in its top DENSE_LOOKUP (the test run keeps k=20, so keep this <= 20)
DENSE_K = 10
DENSE_LOOKUP = 20     # drop candidate pairs whose best blocking similarity is below this

# ---- decision ----
ONE_TO_ONE = True  # each S2/S3 record may match at most one S1 entity (verify with eda.py)
BETA2 = 0.25       # F0.5 -> beta^2

LGB_PARAMS = dict(
    objective="binary",
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=40,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    verbose=-1,
    seed=SEED,
    num_threads=N_JOBS,
)
