import os
INPUT_FOLDER = "./dataset"
OUTPUT_DIR = "."
OUTPUT_FILE = "./positional_index.txt"

SPARK_APP_NAME = "PositionalIndex"
SPARK_MASTER = "local[*]"

from positional_index import initialize_spark, build_positional_index
from tfidf import compute_tf, compute_idf, compute_tfidf, compute_normalized_tfidf
from search import search_query

if __name__ == "__main__":
    sc = initialize_spark(SPARK_APP_NAME, SPARK_MASTER)
    
    positional_index_results, positional_index_dict, all_docs = build_positional_index(
        sc, INPUT_FOLDER, OUTPUT_FILE
    )

    tf_matrix = compute_tf(positional_index_dict, all_docs, OUTPUT_DIR)
    
    num_docs = len(all_docs)
    idf = compute_idf(positional_index_dict, num_docs, OUTPUT_DIR)
    
    tfidf_matrix = compute_tfidf(tf_matrix, idf, all_docs, OUTPUT_DIR)
    
    normalized_tfidf, doc_lengths = compute_normalized_tfidf(
        tfidf_matrix, all_docs, OUTPUT_DIR
    )
    
    # Example query
    query = '"fools fear" AND "in"'
    search_query(query, positional_index_dict, idf, normalized_tfidf)
    
    print("\n" + "="*80)
    print("ALL CALCULATIONS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nDone!")
    
    sc.stop()