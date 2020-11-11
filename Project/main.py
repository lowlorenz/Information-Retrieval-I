from ImageHandler import *
from Utils import *
import pandas as pd
from skimage.feature import ORB
import itertools

handler = ImageHandler()

map_descriptors = extract_map_ORB_c(handler)

clusters, centroids = cluster_k_means(map_descriptors, 100)

#query_descriptors = extract_query_ORB_c(handler)

map_bow_euc = bag_of_words_matrix_c_euclidian(centroids, handler.map_images_gray, distance=euclidian)
map_bow_man = bag_of_words_matrix_c_manhatten(centroids, handler.map_images_gray, distance=manhatten)

euclidian_run = {
    'query_index': [],
    'map_index' : [],
    'similar'   : [],
    'distance'  : []
}

# N = len(handler.query_images_gray)
# for index in range(N):
#     print(f'logging: {index}/{N}')
#     query_orb = extract_ORB(handler.query_images_gray[index])
#     query_bow = bag_of_words(centroids, query_orb)
#     most_similar, distances = retrieve_images(map_bow_euc, query_bow, euclidian)
#     euclidian_run['query_index'].append([index] *  N)
#     euclidian_run['map_index'].append(most_similar)
#     euclidian_run['similar'].append([handler.similarity(index, map_index) for map_index in most_similar])
#     euclidian_run['distance'].append(distances)


# for k in euclidian_run.keys():
#     euclidian_run[k] = list(itertools.chain(*euclidian_run[k]))

# df = pd.DataFrame (euclidian_run, columns = euclidian_run.keys())
# # df.to_excel()


########## SIFT & Kmeans  ##############################################################

handler_cv = ImageHandler(method='opencv')

sift = cv2.SIFT_create(nfeatures=200) # May return more than specified number of features if there are ties.
n_features_sift = extract_SIFT(handler_cv.map_images_gray[0], sift).shape[0]

map_descriptors_sift = extract_map_SIFT_c(handler_cv, sift)

clusters_sift, centroids_sift = cluster_k_means(map_descriptors_sift, 100)

n_images = len(handler_cv.map_images_gray)
map_bow_sift = bag_of_words_sift_all_c(centroids_sift, map_descriptors_sift, n_images, n_features_sift)

# map_bow_man = bag_of_words_matrix_c_manhatten(centroids, handler.map_images_gray, distance=manhatten)

sift_run = {
    'query_index': [],
    'map_index' : [],
    'similar'   : [],
    'distance'  : []
}

N = len(handler_cv.query_images_gray)
for index in range(N):
    print(f'logging: {index}/{N}')
    query_sift = extract_SIFT(handler_cv.query_images_gray[index], sift)
    query_bow_sift = bag_of_words(centroids_sift, query_sift)
    most_similar_sift, distances_sift = retrieve_images(map_bow_sift, query_bow_sift, euclidian)
    sift_run['query_index'].append([index] *  N)
    sift_run['map_index'].append(most_similar_sift)
    sift_run['similar'].append([handler_cv.similarity(index, map_index) for map_index in most_similar_sift])
    sift_run['distance'].append(distances_sift)

for k in sift_run.keys():
    sift_run[k] = list(itertools.chain(*sift_run[k]))

df = pd.DataFrame (sift_run, columns = sift_run.keys())
# df.to_excel()

########## SIFT & GMM  ##############################################################
# The initial sift data from the section above will be re-used here.

n_components=100
gmm = gaussian_mixture_model(map_descriptors_sift, n_components=n_components) # Fit the model to find the clusters/components.

n_images = len(handler_cv.map_images_gray)
map_bow_gmm = bag_of_words_gmm_all_c(gmm, map_descriptors_sift, n_images, n_features_sift)

gmm_run = {
    'query_index': [],
    'map_index' : [],
    'similar'   : [],
    'distance'  : []
}

N = len(handler_cv.query_images_gray)
print("Retrieving using SIFT & GMM...")
for index in range(N):
    print(f'logging: {index}/{N}')
    query_sift = extract_SIFT(handler_cv.query_images_gray[index], sift)
    query_bow_gmm = bag_of_words_gmm(gmm, query_sift)
    most_similar_gmm, distances_gmm = retrieve_images(map_bow_gmm, query_bow_gmm, euclidian)
    gmm_run['query_index'].append([index] *  N)
    gmm_run['map_index'].append(most_similar_gmm)
    gmm_run['similar'].append([handler.similarity(index, map_index) for map_index in most_similar_gmm])
    gmm_run['distance'].append(distances_gmm)

for k in gmm_run.keys():
    gmm_run[k] = list(itertools.chain(*gmm_run[k]))

df = pd.DataFrame(gmm_run, columns = gmm_run.keys())
# df.to_excel()