import os
import math

os.environ['PYSPARK_SUBMIT_ARGS'] = '--conf spark.driver.extraJavaOptions="-Djava.security.manager=allow" --conf spark.executor.extraJavaOptions="-Djava.security.manager=allow" pyspark-shell'

def compute_tf(positional_index, all_docs, output_dir):
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
    
    # Display TF matrix
    print("\nTerm Frequency Matrix:")
    print("-" * 90)
    print(f"{'Term':<15}", end="")
    for doc in all_docs:
        print(f"{doc:<8}", end="")
    print()
    print("-" * 90)
    
    for term in sorted(tf_matrix.keys()):
        print(f"{term:<15}", end="")
        for doc in all_docs:
            print(f"{tf_matrix[term][doc]:<8}", end="")
        print()
    
    # Save to file
    output_file = os.path.join(output_dir, "term_frequency.txt")
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
    
    print(f"\nTerm frequency matrix saved to: {output_file}")
    
    return tf_matrix

def compute_idf(positional_index, num_docs, output_dir):
    """Compute Inverse Document Frequency for each term"""
    print("\n" + "="*80)
    print("COMPUTING INVERSE DOCUMENT FREQUENCY (IDF)")
    print("="*80)
    
    idf = {}
    df_values = {}
    for term in positional_index:
        df = len(positional_index[term])
        df_values[term] = df
        idf[term] = math.log10(num_docs / df)
    
    # Display IDF
    print("\nIDF Values:")
    print("-" * 80)
    print(f"{'Term':<20}{'DF':<10}{'IDF':<15}")
    print("-" * 80)
    
    for term in sorted(idf.keys()):
        print(f"{term:<20}{df_values[term]:<10}{idf[term]:<15.6f}")
    
    # Save to file
    output_file = os.path.join(output_dir, "idf.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{'Term':<20}{'IDF':<15}\n")
        f.write("-" * 80 + "\n")
        for term in sorted(idf.keys()):
            f.write(f"{term:<20}{idf[term]:<15.6f}\n")
    
    print(f"\nIDF values saved to: {output_file}")
    
    return idf

def compute_tfidf(tf_matrix, idf, all_docs, output_dir):
    """Compute TF-IDF matrix"""
    print("\n" + "="*80)
    print("COMPUTING TF-IDF MATRIX")
    print("="*80)
    
    tfidf_matrix = {}
    for term in tf_matrix:
        tfidf_matrix[term] = {}
        for doc in all_docs:
            tfidf_matrix[term][doc] = tf_matrix[term][doc] * idf[term]
    
    # Display TF-IDF matrix
    print("\nTF-IDF Matrix:")
    print("-" * 130)
    print(f"{'Term':<15}", end="")
    for doc in all_docs:
        print(f"{doc:<12}", end="")
    print()
    print("-" * 130)
    
    for term in sorted(tfidf_matrix.keys()):
        print(f"{term:<15}", end="")
        for doc in all_docs:
            print(f"{tfidf_matrix[term][doc]:<12.6f}", end="")
        print()
    
    # Save to file
    output_file = os.path.join(output_dir, "tfidf.txt")
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
    
    print(f"\nTF-IDF matrix saved to: {output_file}")
    
    return tfidf_matrix

def compute_normalized_tfidf(tfidf_matrix, all_docs, output_dir):
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
    output_file = os.path.join(output_dir, "normalized_tfidf.txt")
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
    
    print(f"\nNormalized TF-IDF matrix saved to: {output_file}")
    
    return normalized_tfidf, doc_lengths