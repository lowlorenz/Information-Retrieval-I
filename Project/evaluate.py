import pandas as pd
from evaluation_metrics import precision_at_k, mean_average_precision, sign_test_values


df = pd.read_excel('runs/Run1.xlsx')
threshold = 22

TP = df[(df.distance < threshold) & (df.similar == 1)].shape[0]
FP = df[(df.distance < threshold) & (df.similar == 0)].shape[0]
FN = df[(df.distance >= threshold) & (df.similar == 1)].shape[0]
TN = df[(df.distance >= threshold) & (df.similar == 0)].shape[0]

print(f'Overall positives: {TP + FN}')
print(f'Overall negatives: {TN + FP}')

print(f'Confusion matrix:')
print(f'{TP:<5} | {FN:<5}\n{FP:<5} | {TN:<5}')


# Find queries with relevant images.
queries_with_relevant = df.groupby('query_index').filter(lambda x: x['similar'].max() == 1)
query_indices = set(queries_with_relevant['query_index'].tolist())

all_relevant = {}
all_retrieved = {}
precision_at_10 = {}

for ind in query_indices:
    retrieved = df[df['query_index'] == ind].map_index.tolist()
    relevant = df[(df['query_index'] == ind) & (df['similar'] == 1)].map_index.tolist()
    
    all_relevant[ind] = relevant
    all_retrieved[ind] = retrieved
    
    precision_at_10[ind] = precision_at_k(relevant, retrieved, 10)

print(mean_average_precision(all_relevant, all_retrieved))
print("Precision at 10:", precision_at_10)
