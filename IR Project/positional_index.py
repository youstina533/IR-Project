from pyspark import SparkContext, SparkConf
import os
import re
import os
# Set Java options to fix compatibility with Java 17+
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
    
    return results
if __name__ == "__main__":
    INPUT_FOLDER = "./dataset" 
    OUTPUT_FILE = "./positional_index.txt"  
    
    # Call the function to build the positional index
    positional_index = build_positional_index(INPUT_FOLDER, OUTPUT_FILE)
    
    # Stop Spark context (clean up resources)
    sc.stop()
    # Always stop Spark when done to free up memory and resources
    
    print("\n Done!")