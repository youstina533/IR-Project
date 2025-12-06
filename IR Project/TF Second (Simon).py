import os
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
    
    all_docs = sorted(list(all_docs), key=lambda x: int(x.replace('doc', '')))
    
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
    
    for term in sorted(tf_matrix.keys()):
        print(f"{term:<15}", end="")
        for doc in all_docs:
            print(f"{tf_matrix[term][doc]:<8}", end="")
        print()
    
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
    
    return tf_matrix

def compute_idf(positional_index, num_docs, OUTPUT_DIR):
    """Compute inverse document frequency for each term"""
    print("\n" + "="*80)
    print("COMPUTING INVERSE DOCUMENT FREQUENCY (IDF)")
    print("="*80)
    
    idf = {}
    df_values = {}
    for term in positional_index:
        df = len(positional_index[term])
        df_values[term] = df
        idf[term] = math.log10(num_docs / df)
    
    # Display IDF (first 20 terms)
    print("\nIDF Values:")
    print("-" * 80)
    print(f"{'Term':<20}{'DF':<10}{'IDF':<15}")
    print("-" * 80)
    
    for term in sorted(idf.keys()):
        print(f"{term:<20}{df_values[term]:<10}{idf[term]:<15.6f}")
    
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
    
    for term in sorted(tfidf_matrix.keys()):
        print(f"{term:<15}", end="")
        for doc in all_docs:
            print(f"{tfidf_matrix[term][doc]:<12.6f}", end="")
        print()
    
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

def compute_normalized_tfidf(tfidf_matrix, all_docs, OUTPUT_DIR):
    """Compute document lengths and normalized TF-IDF"""
    print("\n" + "="*80)
    print("COMPUTING DOCUMENT LENGTHS AND NORMALIZED TF-IDF")
    print("="*80)
    
    # Calculate document lengths
    doc_lengths = {}
    for doc in all_docs:
        length = 0
        for term in tfidf_matrix:
            length += tfidf_matrix[term][doc] ** 2
        doc_lengths[doc] = math.sqrt(length)
    
    # Display document lengths
    print("\nDocument Lengths:")
    print("-" * 80)
    for doc in all_docs:
        print(f"{doc} length {doc_lengths[doc]:.6f}")
    
    # Compute Normalized TF-IDF
    normalized_tfidf = {}
    for term in tfidf_matrix:
        normalized_tfidf[term] = {}
        for doc in all_docs:
            if doc_lengths[doc] > 0:
                normalized_tfidf[term][doc] = tfidf_matrix[term][doc] / doc_lengths[doc]
            else:
                normalized_tfidf[term][doc] = 0
    
    # Display normalized TF-IDF matrix
    print("\nNormalized TF-IDF:")
    print("-" * 130)
    print(f"{'Term':<15}", end="")
    for doc in all_docs:
        print(f"{doc:<12}", end="")
    print()
    print("-" * 130)
    
    for term in sorted(normalized_tfidf.keys()):
        print(f"{term:<15}", end="")
        for doc in all_docs:
            print(f"{normalized_tfidf[term][doc]:<12.6f}", end="")
        print()
    
    # Save to file
    output_file = os.path.join(OUTPUT_DIR, "normalized_tfidf.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Document Lengths:\n")
        f.write("-" * 80 + "\n")
        for doc in all_docs:
            f.write(f"{doc} length {doc_lengths[doc]:.6f}\n")
        
        f.write("\n")
        f.write(f"{'Term':<15}")
        for doc in all_docs:
            f.write(f"{doc:<12}")
        f.write("\n")
        f.write("-" * 120 + "\n")
        for term in sorted(normalized_tfidf.keys()):
            f.write(f"{term:<15}")
            for doc in all_docs:
                f.write(f"{normalized_tfidf[term][doc]:<12.6f}")
            f.write("\n")
    
    print(f"\n[OK] Normalized TF-IDF matrix saved to: {output_file}")
    
    return normalized_tfidf, doc_lengths

def parse_query(query):
    """Parse query into phrases + operators (supports AND / AND NOT)."""
    query = query.strip()
    phrases = re.findall(r'"(.*?)"', query)

    operator = None
    if "AND NOT" in query:
        operator = "AND NOT"
    elif "AND" in query:
        operator = "AND"

    return phrases, operator

def search_phrase(phrase, positional_index):
    """Search for a phrase using the positional index."""
    words = phrase.lower().split()
    if len(words) == 1:
        return set(positional_index.get(words[0], {}).keys())

    first_word = words[0]

    if first_word not in positional_index:
        return set()

    candidate_docs = set(positional_index[first_word].keys())
    result_docs = set()

    for doc in candidate_docs:
        first_positions = positional_index[first_word][doc]
        match_found = False

        for pos in first_positions:
            flag = True
            for i in range(1, len(words)):
                w = words[i]
                if w not in positional_index or doc not in positional_index[w]:
                    flag = False
                    break
                if (pos + i) not in positional_index[w][doc]:
                    flag = False
                    break

            if flag:
                match_found = True
                break

        if match_found:
            result_docs.add(doc)
        
    return result_docs

def apply_boolean_logic(docs1, docs2, operator):
    if operator == "AND":
        return docs1.intersection(docs2)
    elif operator == "AND NOT":
        return docs1 - docs2
    else:
        return docs1

def build_query_vector(query_terms, idf):
    """Build vector for the query using TF-IDF (Raw TF * IDF)"""
    query_tf = {}
    for t in query_terms:
        query_tf[t] = query_tf.get(t, 0) + 1

    query_vec = {}
    for t in query_tf:
        if t in idf:
            query_vec[t] = query_tf[t] * idf[t]
    return query_vec, query_tf

def cosine_similarity_normalized(query_vec, query_length, doc, normalized_tfidf):
    """Compute cosine similarity using normalized TF-IDF"""
    dot = 0
    for term in query_vec:
        if term in normalized_tfidf and doc in normalized_tfidf[term]:
            # query_vec[term] is already normalized by query_length in the calling function
            dot += (query_vec[term] / query_length) * normalized_tfidf[term][doc]
    
    return dot

def search_query(query, positional_index, idf, normalized_tfidf):
    """Main function for full query search + ranking using normalized TF-IDF"""

    print("\n" + "="*80)
    print(f"PROCESSING QUERY: {query}")
    print("="*80)

    phrases, operator = parse_query(query)
    print(f"\nDetected phrases: {phrases}")
    print(f"Operator: {operator}")

    # Step 1: Search first phrase
    result_docs = search_phrase(phrases[0], positional_index)
    print(f"\nInitial matched docs for '{phrases[0]}': {result_docs}")

    # Step 2: Boolean + second phrase (if exists)
    if operator and len(phrases) > 1:
        docs_second = search_phrase(phrases[1], positional_index)
        print(f"Docs for second phrase '{phrases[1]}': {docs_second}")
        result_docs = apply_boolean_logic(result_docs, docs_second, operator)

    print(f"\nFinal matched documents: {result_docs}")

    # Step 3: Compute Query Vector
    query_terms = []
    for p in phrases:
        query_terms.extend(p.lower().split())

    query_vec, query_tf = build_query_vector(query_terms, idf)
    
    # Calculate query length
    query_length = math.sqrt(sum(v*v for v in query_vec.values()))

    # Display query information
    print("\n" + "-"*80)
    print("QUERY VECTOR INFORMATION:")
    print("-" * 80)
    print(f"{'Term':<15}{'TF-raw':<10}{'IDF':<15}{'TF*IDF':<15}")
    print("-" * 80)
    for term in sorted(query_vec.keys()):
        print(f"{term:<15}{query_tf[term]:<10}{idf[term]:<15.4f}{query_vec[term]:<15.4f}")
    print(f"\nQuery Length: {query_length:.6f}")

    # Step 4: Compute Similarity Scores
    scores = {}
    for doc in result_docs:
        scores[doc] = cosine_similarity_normalized(query_vec, query_length, doc, normalized_tfidf)

    # Step 5: Ranking
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "-"*80)
    print("RANKED RESULTS:")
    print("-" * 80)
    for doc, score in ranked_docs:
        print(f"{doc:<10}  similarity = {score:.6f}")
    
    print("\nReturned documents:", ", ".join([doc for doc, score in ranked_docs]))

    return ranked_docs

if __name__ == "__main__":
    INPUT_FOLDER = "./datase" 
    OUTPUT_FILE = "./positional_index.txt"
    OUTPUT_DIR = "."  # Output directory for TF-IDF files
    
    # Call the function to build the positional index
    positional_index_results, positional_index_dict, all_docs = build_positional_index(INPUT_FOLDER, OUTPUT_FILE)
    
    # Compute TF (Raw counts)
    tf_matrix = compute_tf(positional_index_dict, all_docs, OUTPUT_DIR) 

    # Compute IDF (Inverse Document Frequency)
    num_docs = len(all_docs)
    idf = compute_idf(positional_index_dict, num_docs, OUTPUT_DIR)
    
    # Compute TF-IDF (Raw TF * IDF)
    tfidf_matrix = compute_tfidf(tf_matrix, idf, all_docs, OUTPUT_DIR)
    
    # Compute Normalized TF-IDF
    normalized_tfidf, doc_lengths = compute_normalized_tfidf(tfidf_matrix, all_docs, OUTPUT_DIR)

    # Example query
    query = '"fools fear" AND "in"'
    search_query(query, positional_index_dict, idf, normalized_tfidf)


    # Stop Spark context (clean up resources)
    sc.stop()
    
    print("\n" + "="*80)
    print("ALL CALCULATIONS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\n Done!")