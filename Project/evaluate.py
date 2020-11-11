import pandas as pd
from scipy.stats import binom_test
from evaluation_metrics import precision_at_10, mean_average_precision, sign_test_values
from matplotlib import pyplot as plt

df_orb = pd.read_excel('runs/run1_orb.xlsx')
df_sift = pd.read_excel('runs/run1_sift.xlsx')
df_gmm = pd.read_excel('runs/run1_gmm.xlsx')
df_gmm_un = pd.read_excel('runs/run1_gmm_unnormalised.xlsx')

threshold = 22

TP = df_orb[(df_orb.distance < threshold) & (df_orb.similar == 1)].shape[0]
FP = df_orb[(df_orb.distance < threshold) & (df_orb.similar == 0)].shape[0]
FN = df_orb[(df_orb.distance >= threshold) & (df_orb.similar == 1)].shape[0]
TN = df_orb[(df_orb.distance >= threshold) & (df_orb.similar == 0)].shape[0]

print(f'Overall positives: {TP + FN}')
print(f'Overall negatives: {TN + FP}')

print(f'Confusion matrix:')
print(f'{TP:<5} | {FN:<5}\n{FP:<5} | {TN:<5}')


# Find queries with relevant images.
queries_with_relevant = df_orb.groupby('query_index').filter(lambda x: x['similar'].max() == 1)
query_indices = set(queries_with_relevant['query_index'].tolist())

all_relevant = {}
all_retrieved_orb = {}
all_retrieved_sift = {}
all_retrieved_gmm = {}
all_retrieved_gmm_un = {}

precision_at_10_orb = {}
precision_at_10_sift = {}
precision_at_10_gmm = {}
precision_at_10_gmm_un = {}


for ind in query_indices:
    relevant = df_orb[(df_orb['query_index'] == ind) & (df_orb['similar'] == 1)].map_index.tolist()
    
    retrieved_orb = df_orb[df_orb['query_index'] == ind].map_index.tolist()
    retrieved_sift = df_sift[df_sift['query_index'] == ind].map_index.tolist()
    retrieved_gmm = df_gmm[df_gmm['query_index'] == ind].map_index.tolist()
    retrieved_gmm_un = df_gmm_un[df_gmm_un['query_index'] == ind].map_index.tolist()
    
    all_relevant[ind] = relevant
    
    all_retrieved_orb[ind] = retrieved_orb
    all_retrieved_sift[ind] = retrieved_sift
    all_retrieved_gmm[ind] = retrieved_gmm
    all_retrieved_gmm_un[ind] = retrieved_gmm_un

    precision_at_10_orb[ind] = precision_at_10(relevant, retrieved_orb)
    precision_at_10_sift[ind] = precision_at_10(relevant, retrieved_sift)
    precision_at_10_gmm[ind] = precision_at_10(relevant, retrieved_gmm)
    precision_at_10_gmm_un[ind] = precision_at_10(relevant, retrieved_gmm_un)


print("\nMAP ORB:", round(mean_average_precision(all_relevant, all_retrieved_orb), 4))
print("MAP SIFT:", round(mean_average_precision(all_relevant, all_retrieved_sift), 4))
print("MAP GMM:", round(mean_average_precision(all_relevant, all_retrieved_gmm), 4))
print("MAP GMM2:", round(mean_average_precision(all_relevant, all_retrieved_gmm_un), 4))


mean_prec_at_10_orb = sum(precision_at_10_orb.values())/len(precision_at_10_orb)
mean_prec_at_10_sift = sum(precision_at_10_sift.values())/len(precision_at_10_sift)
mean_prec_at_10_gmm = sum(precision_at_10_gmm.values())/len(precision_at_10_gmm)
mean_prec_at_10_gmm_un = sum(precision_at_10_gmm_un.values())/len(precision_at_10_gmm_un)


print("Mean Precision at 10 ORB:", round(mean_prec_at_10_orb,4))
print("Mean Precision at 10 SIFT:", round(mean_prec_at_10_sift,4))
print("Mean Precision at 10 GMM:", round(mean_prec_at_10_gmm,4))
print("Mean Precision at 10 GMM2:", round(mean_prec_at_10_gmm_un,4))

# print(precision_at_10_orb)
# print(precision_at_10_sift)
# print(precision_at_10_gmm)

# st is a tuple, the number of queries where GMM is better and the number where GMM is worse.
st_orb_gmm = sign_test_values(precision_at_10, all_relevant, all_retrieved_orb, all_retrieved_gmm)
st_orb_gmm2 = sign_test_values(precision_at_10, all_relevant, all_retrieved_orb, all_retrieved_gmm_un)
st_orb_sift = sign_test_values(precision_at_10, all_relevant, all_retrieved_orb, all_retrieved_sift)
st_kmeans_gmm = sign_test_values(precision_at_10, all_relevant, all_retrieved_sift, all_retrieved_gmm)

print("Binomial Test ORB-Kmeans vs. SIFT-GMM: p =", round(binom_test(st_orb_gmm),3), "  |  better,worse = ", st_orb_gmm) 
print("Binomial Test ORB-Kmeans vs. SIFT-GMM2: p =", round(binom_test(st_orb_gmm2),3), "  |  better,worse = ", st_orb_gmm2)  
print("Binomial Test ORB-Kmeans vs. SIFT-Kmeans: p =", round(binom_test(st_orb_sift),3), "  |  better,worse = ", st_orb_sift)  
print("Binomial Test SIFT-Kmeans vs. SIFT-GMM : p =", round(binom_test(st_kmeans_gmm),3), "  |  better,worse = ", st_kmeans_gmm)  


# Compare histograms of the precision_at_10 for each system.
fig, axes = plt.subplots(4, 1, sharex = True, sharey = True)
axes[0].hist(list(precision_at_10_orb.values()), bins=20, range=(0,1))
axes[1].hist(list(precision_at_10_sift.values()), bins=20, range=(0,1))
axes[2].hist(list(precision_at_10_gmm.values()), bins=20, range=(0,1))
axes[3].hist(list(precision_at_10_gmm_un.values()), bins=20, range=(0,1))

axes[0].title.set_text('ORB-Kmeans')
axes[1].title.set_text('SIFT-Kmeans')
axes[2].title.set_text('SIFT-GMM')
axes[3].title.set_text('SIFT-GMM (Unnormalised)')
axes[1].set_ylabel('Number of Queries')
axes[2].set_xlabel('Precision_at_10')
plt.tight_layout()
plt.savefig('Precision_at_10_hist.png')
