from ImageHandler import *
from Utils import *
from skimage.feature import ORB

handler = ImageHandler()

map_descriptors = extract_map_ORB_c(handler)

clusters, centroids = cluster_k_means(map_descriptors)

#query_descriptors = extract_query_ORB_c(handler)

map_bow_euc = bag_of_words_matrix_c_euclidian(centroids, handler.map_images_gray, distance=euclidian)
map_bow_man = bag_of_words_matrix_c_manhatten(centroids, handler.map_images_gray, distance=manhatten)


for index in range(len(handler.query_images_gray)):
    query_orb = extract_ORB(handler.query_images_gray[0])
    query_bow = bag_of_words(centroids, query_orb)
    most_similar, distances = retrieve_images(map_bow_euc, query_bow, euclidian)
    

