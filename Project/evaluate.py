import pandas as pd

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
