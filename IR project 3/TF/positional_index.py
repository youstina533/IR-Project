import os
import re

os.environ['PYSPARK_SUBMIT_ARGS'] = '--conf spark.driver.extraJavaOptions="-Djava.security.manager=allow" --conf spark.executor.extraJavaOptions="-Djava.security.manager=allow" pyspark-shell'

def extract_terms_with_positions(doc_tuple):
    """Extract terms with their positions from a document"""
    doc_name, content = doc_tuple
    doc_number = doc_name.replace('.txt', '')
    formatted_doc = f"doc{doc_number}"
    words = re.findall(r'\b\w+\b', content.lower())
    
    result = []
    for position, term in enumerate(words, start=1):
        result.append((term, formatted_doc, position))
    return result

def format_output(term_docs):
    """Format the positional index output"""
    term, doc_list = term_docs
    doc_list_sorted = sorted(doc_list, key=lambda x: x[0])
    
    doc_strings = []
    for doc_name, positions in doc_list_sorted:
        pos_str = ', '.join(map(str, positions))
        doc_strings.append(f"{doc_name}: {pos_str}")
    
    return f"{term} {'; '.join(doc_strings)}"

def build_positional_index(sc, input_folder, output_file):
    """Build positional index from documents"""
    print("\n" + "="*80)
    print("BUILDING POSITIONAL INDEX")
    print("="*80)
    
    # Get list of all .txt files
    files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    
    # Read documents
    documents = sc.parallelize([
        (filename, open(os.path.join(input_folder, filename), 'r').read())
        for filename in files
    ])
    
    # Extract terms with positions
    term_doc_pos = documents.flatMap(extract_terms_with_positions)
    
    # Transform to ((term, doc), position)
    term_doc_with_pos = term_doc_pos.map(lambda x: ((x[0], x[1]), x[2]))
    
    # Group positions by (term, document)
    grouped_by_term_doc = term_doc_with_pos.groupByKey().mapValues(list)
    
    # Transform to (term, (doc, [positions]))
    term_with_doc_positions = grouped_by_term_doc.map(
        lambda x: (x[0][0], (x[0][1], sorted(x[1])))
    )
    
    # Group all documents by term
    positional_index = term_with_doc_positions.groupByKey().mapValues(list)
    
    # Format and sort output
    formatted_index = positional_index.map(format_output)
    sorted_index = formatted_index.sortBy(lambda x: x.split()[0])
    
    # Collect results
    results = sorted_index.collect()
    
    # Write to output file
    with open(output_file, 'w') as f:
        for line in results:
            f.write(line + '\n')
    
    # Print statistics
    print(f"\nPositional index created successfully!")
    print(f"Total unique terms: {len(results)}")
    print(f"Output saved to: {output_file}")
    
    # Display sample
    print("\n=== Sample Output (first 10 terms) ===")
    for i, line in enumerate(results[:10]):
        print(line)
    
    # Create dictionary for TF-IDF calculations
    positional_index_dict = {}
    all_docs = set()
    
    for term, doc_list in positional_index.collect():
        positional_index_dict[term] = {}
        for doc_name, positions in doc_list:
            positional_index_dict[term][doc_name] = positions
            all_docs.add(doc_name)
    
    all_docs = sorted(list(all_docs), key=lambda x: int(x.replace('doc', '')))
    
    return results, positional_index_dict, all_docs