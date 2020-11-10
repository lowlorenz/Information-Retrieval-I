from ImageHandler import *
from Utils import *
from skimage.feature import ORB

handler = ImageHandler()
#plot([handler.query_images_gray[0]], cols=1)

map_descriptors = extract_map_ORB(handler)

clusters, centroids = cluster_k_means(map_descriptors)

query_descriptors = extract_query_ORB(handler)

print('calculate bow\'s')
bow = bag_of_words_matrix(centroids, query_descriptors)
print(bow.shape)
