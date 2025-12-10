from pyspark import SparkContext, SparkConf

INPUT_FOLDER = "..\project_dataSet"
OUTPUT_DIR = "."
OUTPUT_FILE = "./positional_index.txt"

SPARK_APP_NAME = "PositionalIndex"
SPARK_MASTER = "local[*]"

from positional_index import build_positional_index
from tfidf import compute_tf, compute_idf, compute_tfidf, compute_normalized_tfidf
from search import search_query

if __name__ == "__main__":
    conf = SparkConf().setAppName(SPARK_APP_NAME).setMaster(SPARK_MASTER)
    sc = SparkContext(conf=conf)
    
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
    while True:
        print("\n" + "-"*80)
        query = input("\nEnter your query: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("\nExiting search engine. Goodbye!")
            break
        
        if not query:
            print("Please enter a valid query.")
            continue
        
        try:
            search_query(query, positional_index_dict, idf, normalized_tfidf)
        except Exception as e:
            print(f"\nError processing query: {e}")
            print("Please check your query format and try again.")
    
    sc.stop()