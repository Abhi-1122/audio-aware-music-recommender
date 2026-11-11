"""Two-stage music recommender on Yambda-50M (retrieve -> rank), in pipeline order.

data (+ sql/) builds the processed tables and the A / B / C time split; metrics holds the evaluation
(Recall, NDCG, Coverage, candidate recall) in the "all" and "discovery" views; sources implements the
non-neural retrievers (popularity, repeat, ALS, audio profile); twotower trains the two-tower model;
retrieval runs every source with FAISS and merges up to 500 candidates per user (stage 1); ranker
re-ranks them with LightGBM LambdaRank (stage 2); serve wraps both stages in a FastAPI prototype.
"""
