import os


# ==================================================================================================
# DATA INGESTION
# ==================================================================================================
RAW_DIR = "artifacts/raw"

CONFIG_PATH =  "config/config.yaml"



# ==================================================================================================
# DATA PROCESSING
# ==================================================================================================
PROCESSED_DIR = "artifacts/processed"
ANIMELIST_CSV = "/home/damieniscreating/anime_suggestion/artifacts/raw/animelist.csv"
ANIME_CSV = "/home/damieniscreating/anime_suggestion/artifacts/raw/anime.csv"
ANIMESYNOPSIS_CSV = "/home/damieniscreating/anime_suggestion/artifacts/raw/anime_with_synopsis.csv"

X_TRAIN_ARRAY = os.path.join(PROCESSED_DIR, 'X_train_array.pkl')
X_TEST_ARRAY = os.path.join(PROCESSED_DIR, 'X_test_array.pkl')
Y_TRAIN = os.path.join(PROCESSED_DIR, 'y_train.pkl')
Y_TEST = os.path.join(PROCESSED_DIR, 'y_test.pkl')

RATING_DF = os.path.join(PROCESSED_DIR, 'rating_df.csv')
DF_PATH = os.path.join(PROCESSED_DIR, 'anime_df.csv')
SYNOPTIS_DF = os.path.join(PROCESSED_DIR, 'synopsis_df.csv')


# ==================================================================================================
# MODEL TRAINING
# ==================================================================================================
MODEL_DIR = "artifacts/model"
WEIGHT_DIR = "artifacts/weights"

MODEL_PATH = os.path.join(MODEL_DIR, 'model.h5')
ANIME_WEIGHTS_PATH = os.path.join(WEIGHT_DIR, 'anime_weights.pkl')
USER_WEIGHTS_PATH = os.path.join(WEIGHT_DIR, 'user_weights.pkl')

CHECK_POINT_FILE_PATH = "artifacts/model_checkpoint/weights.weights.h5"