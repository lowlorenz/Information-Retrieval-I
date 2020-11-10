from ImageHandler import *
from Utils import *
from skimage.feature import ORB

handler = ImageHandler()
#plot([handler.query_images_gray[0]], cols=1)

map_descriptors = extract_map_ORB_c(handler)

clusters, centroids = cluster_k_means(map_descriptors)

query_descriptors = extract_query_ORB_c(handler)

#print(f'query_descriptors: {query_descriptors.shape}')
bow = bag_of_words_matrix_c(centroids, handler.map_images_gray)
print(bow.shape)
