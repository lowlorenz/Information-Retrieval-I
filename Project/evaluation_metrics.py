# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:49:15 2020

@author: Ricky
"""

def precision(relevant, retrieved):
    if len(retrieved) == 0:
        return 0
    
    if len(relevant) == 0:
        return 1
    
    relevant_and_retrieved = [k for k in retrieved if k in relevant]
    return len(relevant_and_retrieved) / len(retrieved)

def precision_at_k(relevant, retrieved, k):
    return precision(relevant, retrieved[:k])

def average_precision(relevant, retrieved):
    # Initialise list of precisions with a zero for each relevant document.
    P = [0] * len(relevant)
    
    for i, doc in enumerate(relevant):
        # If a relevant document is not retrieved, the precision value is taken to be zero. 
        if doc not in retrieved:
            P[i] = 0
        else:
            # Find the precision for the top k documents when doc is retrieved.
            k = retrieved.index(doc)
            P[i] = precision_at_k(relevant, retrieved, k)
    
    # Return the average precision
    return sum(P)/len(P)

def mean_average_precision(all_relevant, all_retrieved):  
    count = len(all_retrieved)
        
    precision_per_query = [average_precision(all_relevant[query], all_retrieved[query])  for query in all_retrieved]
    total = sum(precision_per_query)
    
    return "mean AP: ", total / count

def sign_test_values(measure, qrels_file, run_file_1, run_file_2):
    all_relevant = read_qrels_file(qrels_file)
    all_retrieved_1 = read_run_file(run_file_1)
    all_retrieved_2 = read_run_file(run_file_2)
    better = 0
    worse  = 0
    # BEGIN ANSWER
    
    for query in all_retrieved_1:
        performance_1 = measure(all_relevant[query], all_retrieved_1[query])
        performance_2 = measure(all_relevant[query], all_retrieved_2[query])
        
        if performance_2 > performance_1:
            better += 1
        # Exclude queries with no performance difference between the two methods.
        elif performance_2 < performance_1:
            worse += 1
    
    # END ANSWER
    return(better, worse)
