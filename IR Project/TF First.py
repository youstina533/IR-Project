from pyspark import SparkContext, SparkConf
import os
import re
import math

os.environ['PYSPARK_SUBMIT_ARGS'] = '--conf spark.driver.extraJavaOptions="-Djava.security.manager=allow" --conf spark.executor.extraJavaOptions="-Djava.security.manager=allow" pyspark-shell'

conf = SparkConf().setAppName("PositionalIndex").setMaster("local[*]")
sc = SparkContext(conf=conf)
def build_positional_index(input_folder, output_file):   
    # Get list of all .txt files in the input folder
    files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]

    # Read each file and create (filename, content) pairs
    documents = sc.parallelize([
        (filename, open(os.path.join(input_folder, filename), 'r').read())
        for filename in files
    ])
    # STEP 2: Extract (term, document, position) from each document
    def extract_terms_with_positions(doc_tuple):
        doc_name, content = doc_tuple
        # Convert to lowercase and split into words
        doc_number = doc_name.replace('.txt', '')
        formatted_doc = f"doc{doc_number}"
        words = re.findall(r'\b\w+\b', content.lower())
        # Create (term, doc_name, position) for each word
        result = []
        for position, term in enumerate(words, start=1):
            result.append((term, formatted_doc , position))
        return result 
    
    # Apply extraction to ALL documents
    term_doc_pos = documents.flatMap(extract_terms_with_positions)
    # STEP 3: Transform to ((term, doc), position) for grouping
    term_doc_with_pos = term_doc_pos.map(lambda x: ((x[0], x[1]), x[2]))

    # STEP 4: Group positions by (term, document)
    grouped_by_term_doc = term_doc_with_pos.groupByKey().mapValues(list)
    # STEP 5: Transform to (term, (doc, [positions]))
    term_with_doc_positions = grouped_by_term_doc.map(
        lambda x: (x[0][0], (x[0][1], sorted(x[1])))
    )
   
    # STEP 6: Group all documents by term
    positional_index = term_with_doc_positions.groupByKey().mapValues(list)
    # STEP 7: Format the output as required
    def format_output(term_docs):
        term, doc_list = term_docs
        # Sort documents for consistent output
        doc_list_sorted = sorted(doc_list, key=lambda x: x[0])
        
        # Build the formatted string
        doc_strings = []
        for doc_name, positions in doc_list_sorted:
            pos_str = ', '.join(map(str, positions))
            doc_strings.append(f"{doc_name}: {pos_str}")
        
        return f"{term} {'; '.join(doc_strings)}"
        
    formatted_index = positional_index.map(format_output)
    # STEP 8: Sort by term and save to file
    sorted_index = formatted_index.sortBy(lambda x: x.split()[0])
    # Collect results from Spark to local Python
    results = sorted_index.collect()
    # Write to output file
    with open(output_file, 'w') as f:
        for line in results:
            f.write(line + '\n')
    # Print success message
    print(f"Positional index created successfully!")
    print(f"Total unique terms: {len(results)}")
    print(f"Output saved to: {output_file}")
    
    # Display first 10 entries as sample
    print("\n=== Sample Output (first 10 terms) ===")
    for i, line in enumerate(results[:10]):
        print(line)
    
    # Also return as dictionary for TF-IDF calculations
    positional_index_dict = {}
    all_docs = set()
    
    for term, doc_list in positional_index.collect():
        positional_index_dict[term] = {}
        for doc_name, positions in doc_list:
            positional_index_dict[term][doc_name] = positions
            all_docs.add(doc_name)
    
    all_docs = sorted(list(all_docs))
    
    return results, positional_index_dict, all_docs

def compute_tf(positional_index, all_docs, OUTPUT_DIR):
    """Compute Term Frequency matrix"""
    print("\n" + "="*80)
    print("COMPUTING TERM FREQUENCY (TF)")
    print("="*80)
    
    # Compute TF
    tf_matrix = {}
    for term in positional_index:
        tf_matrix[term] = {}
        for doc in all_docs:
            if doc in positional_index[term]:
                tf_matrix[term][doc] = len(positional_index[term][doc])
            else:
                tf_matrix[term][doc] = 0
    
    # Display TF matrix (first 15 terms)
    print("\nTerm Frequency Matrix:")
    print("-" * 80)
    print(f"{'Term':<15}", end="")
    for doc in all_docs:
        print(f"{doc:<8}", end="")
    print()
    print("-" * 80)
    
    for i, term in enumerate(sorted(tf_matrix.keys())):
        if i >= 15:
            break
        print(f"{term:<15}", end="")
        for doc in all_docs:
            print(f"{tf_matrix[term][doc]:<8}", end="")
        print()
    
    if len(tf_matrix) > 15:
        print(f"... (showing 15 of {len(tf_matrix)} terms)")
    
    # Save to file
    output_file = os.path.join(OUTPUT_DIR, "term_frequency.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{'Term':<15}")
        for doc in all_docs:
            f.write(f"{doc:<8}")
        f.write("\n")
        f.write("-" * 80 + "\n")
        for term in sorted(tf_matrix.keys()):
            f.write(f"{term:<15}")
            for doc in all_docs:
                f.write(f"{tf_matrix[term][doc]:<8}")
            f.write("\n")
    
    print(f"\n[OK] Term frequency matrix saved to: {output_file}")
    
    return tf_matrix, all_docs

def compute_idf(positional_index, num_docs, OUTPUT_DIR):
    """Compute inverse document frequency for each term"""
    print("\n" + "="*80)
    print("COMPUTING INVERSE DOCUMENT FREQUENCY (IDF)")
    print("="*80)
    
    idf = {}
    for term in positional_index:
        df = len(positional_index[term])  # Document frequency
        idf[term] = math.log10(num_docs / df)
    
    # Display IDF (first 20 terms)
    print("\nIDF Values:")
    print("-" * 80)
    print(f"{'Term':<20}{'IDF':<15}")
    print("-" * 80)
    
    for i, term in enumerate(sorted(idf.keys())):
        if i >= 20:
            break
        print(f"{term:<20}{idf[term]:<15.6f}")
    
    if len(idf) > 20:
        print(f"... (showing 20 of {len(idf)} terms)")
    
    # Save to file
    output_file = os.path.join(OUTPUT_DIR, "idf.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{'Term':<20}{'IDF':<15}\n")
        f.write("-" * 80 + "\n")
        for term in sorted(idf.keys()):
            f.write(f"{term:<20}{idf[term]:<15.6f}\n")
    
    print(f"\n[OK] IDF values saved to: {output_file}")
    
    return idf

def compute_tfidf(tf_matrix, idf, all_docs, OUTPUT_DIR):
    """Compute TF-IDF matrix"""
    print("\n" + "="*80)
    print("COMPUTING TF-IDF MATRIX")
    print("="*80)
    
    tfidf_matrix = {}
    for term in tf_matrix:
        tfidf_matrix[term] = {}
        for doc in all_docs:
            tfidf_matrix[term][doc] = tf_matrix[term][doc] * idf[term]
    
    # Display TF-IDF matrix (first 15 terms)
    print("\nTF-IDF Matrix:")
    print("-" * 80)
    print(f"{'Term':<15}", end="")
    for doc in all_docs:
        print(f"{doc:<12}", end="")
    print()
    print("-" * 80)
    
    for i, term in enumerate(sorted(tfidf_matrix.keys())):
        if i >= 15:
            break
        print(f"{term:<15}", end="")
        for doc in all_docs:
            print(f"{tfidf_matrix[term][doc]:<12.6f}", end="")
        print()
    
    if len(tfidf_matrix) > 15:
        print(f"... (showing 15 of {len(tfidf_matrix)} terms)")
    
    # Save to file
    output_file = os.path.join(OUTPUT_DIR, "tfidf.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{'Term':<15}")
        for doc in all_docs:
            f.write(f"{doc:<12}")
        f.write("\n")
        f.write("-" * 80 + "\n")
        for term in sorted(tfidf_matrix.keys()):
            f.write(f"{term:<15}")
            for doc in all_docs:
                f.write(f"{tfidf_matrix[term][doc]:<12.6f}")
            f.write("\n")
    
    print(f"\n[OK] TF-IDF matrix saved to: {output_file}")
    
    return tfidf_matrix
if __name__ == "__main__":
    INPUT_FOLDER = "./dataset" 
    OUTPUT_FILE = "./positional_index.txt"
    OUTPUT_DIR = "."  # Output directory for TF-IDF files
    
    # Call the function to build the positional index
    positional_index_results, positional_index_dict, all_docs = build_positional_index(INPUT_FOLDER, OUTPUT_FILE)
    
    # Compute TF (Term Frequency)
    tf_matrix, all_docs = compute_tf(positional_index_dict, all_docs, OUTPUT_DIR)
    
    # Compute IDF (Inverse Document Frequency)
    num_docs = len(all_docs)
    idf = compute_idf(positional_index_dict, num_docs, OUTPUT_DIR)
    
    # Compute TF-IDF
    tfidf_matrix = compute_tfidf(tf_matrix, idf, all_docs, OUTPUT_DIR)
    
    # Stop Spark context (clean up resources)
    sc.stop()
    # Always stop Spark when done to free up memory and resources
    
    print("\n" + "="*80)
    print("ALL CALCULATIONS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\n Done!")