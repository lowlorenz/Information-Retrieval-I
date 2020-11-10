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

N = len(handler.query_images_gray)
for index in range(N):
    print(f'logging: {index}/{N}')
    query_orb = extract_ORB(handler.query_images_gray[index])
    query_bow = bag_of_words(centroids, query_orb)
    most_similar, distances = retrieve_images(map_bow_euc, query_bow, euclidian)
    euclidian_run['query_index'].append([index] *  N)
    euclidian_run['map_index'].append(most_similar)
    euclidian_run['similar'].append([handler.similarity(index, map_index) for map_index in most_similar])
    euclidian_run['distance'].append(distances)


for k in euclidian_run.keys():
    euclidian_run[k] = list(itertools.chain(*euclidian_run[k]))

df = pd.DataFrame (euclidian_run, columns = euclidian_run.keys())
df.to_excel()