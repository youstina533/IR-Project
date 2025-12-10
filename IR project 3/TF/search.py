import re
import math
 
def parse_query(query):
    query = query.strip()
    phrases = re.findall(r'"(.*?)"', query)

    operator = None
    if "AND NOT" in query:
        operator = "AND NOT"
    elif "AND" in query:
        operator = "AND"

    return phrases, operator

def search_phrase(phrase, positional_index):
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
    query_tf = {}
    for t in query_terms:
        query_tf[t] = query_tf.get(t, 0) + 1

    query_vec = {}
    for t in query_tf:
        if t in idf:
            query_vec[t] = query_tf[t] * idf[t]
    return query_vec, query_tf

def cosine_similarity_normalized(query_vec, query_length, doc, normalized_tfidf):
    dot = 0
    for term in query_vec:
        if term in normalized_tfidf and doc in normalized_tfidf[term]:
            dot += (query_vec[term] / query_length) * normalized_tfidf[term][doc]
    
    return dot

def search_query(query, positional_index, idf, normalized_tfidf):
    phrases, operator = parse_query(query)
    
    if operator == "AND":
        query_terms = []
        for p in phrases:
            query_terms.extend(p.lower().split())
    else:
        query_terms = phrases[0].lower().split()

    result_docs = search_phrase(phrases[0], positional_index)

    if operator and len(phrases) > 1:
        docs_second = search_phrase(phrases[1], positional_index)
        result_docs = apply_boolean_logic(result_docs, docs_second, operator)

    query_vec, query_tf = build_query_vector(query_terms, idf)
    
    query_weighted_tf = {}
    for term in query_tf:
        if query_tf[term] > 0:
            query_weighted_tf[term] = 1 + math.log10(query_tf[term])
        else:
            query_weighted_tf[term] = 0
    
    query_vec = {}
    for term in query_weighted_tf:
        if term in idf:
            query_vec[term] = query_weighted_tf[term] * idf[term]
    
    query_length = math.sqrt(sum(v*v for v in query_vec.values()))

    sorted_docs = sorted(result_docs, key=lambda x: int(x.replace('doc', '')))

    print("\nquery")
    print("" + " ".join(sorted(query_vec.keys())))
    print()
    
    print(f"{'':15}{'tf-raw':<10}{'v tf(1+ log tf)':<15}{'  idf':<10}{'tf*idf':<15}{'normalized':<15}", end="")
    for doc in sorted_docs:
        print(f"{doc:<10}", end="")
    print()
    
    products = {}
    for term in sorted(query_vec.keys()):
        products[term] = {}
        normalized_query_term = query_vec[term] / query_length
        
        for doc in sorted_docs:
            if term in normalized_tfidf and doc in normalized_tfidf[term]:
                products[term][doc] = normalized_query_term * normalized_tfidf[term][doc]
            else:
                products[term][doc] = 0
    
    for term in sorted(query_vec.keys()):
        print(f"{term:<15}", end="")
        print(f"{query_tf[term]:<10}", end="")
        print(f"{query_weighted_tf[term]:<15.4f}", end="")
        print(f"{idf[term]:<10.4f}", end="")
        print(f"{query_vec[term]:<15.4f}", end="")
        print(f"{(query_vec[term]/query_length):<15.5f}", end="")

        for doc in sorted_docs:
            print(f"{products[term][doc]:<10.6f}", end="")
        print()
    
    print(f"\n{'':75}{'sum':<6}", end="")
    for doc in sorted_docs:
        doc_sum = sum(products[term][doc] for term in query_vec.keys())
        print(f"{doc_sum:<10.6f}", end="")
    print()
    
    print(f"\nquery length  {query_length:.6f}")
    
    scores = {}
    for doc in result_docs:
        scores[doc] = cosine_similarity_normalized(query_vec, query_length, doc, normalized_tfidf)
    
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    print()
    for doc in sorted_docs:
        print(f"similarity (q , {doc})   {scores[doc]:.6f}")
    
    print(f"returned docs    {','.join([doc for doc, score in ranked_docs])}")

    return ranked_docs