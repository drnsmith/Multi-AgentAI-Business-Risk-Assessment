# config.py

# Whether to use Maximal Marginal Relevance (MMR) during retrieval
USE_MMR = True

# Threshold for filtering retrieved documents by score (if supported by retriever)
SCORE_THRESHOLD = 0.7

# Max number of documents to retrieve
TOP_K = 5

# Directory where PDF documents are stored
DATA_DIR = "data"

FILENAME_FILTERING_ENABLED = True
# Whether to filter chunks based on filename clues in the query
FILTER_BY_PAGE = True


