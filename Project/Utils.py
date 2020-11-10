
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from skimage.feature import ORB
import os

orb = ORB(n_keypoints=200)

def make_cached(func, filename):
    def inner(*args, **kwargs):        
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                return np.load(f)
        
        result = func(*args, **kwargs)
        with open(filename, 'wb') as f:
            np.save(f, result)

        return result
        
    return inner

def extract_ORB(img):
    orb.detect_and_extract(img)
    return orb.descriptors 

def extract_map_ORB(handler):
    # otherwise extract them and safe them in a npy fil
    return np.vstack( 
        (extract_ORB(img) for img in handler.map_images_gray) 
        )
    
extract_map_ORB = make_cached(extract_map_ORB, 'map_orb.npy')

def extract_query_ORB(handler):
    return np.vstack( (extract_ORB(img) for img in handler.query_images_gray) )
    
extract_query_ORB = make_cached(extract_query_ORB, 'query_orb.npy')

def cluster_k_means(descriptors):
    # clustering
    K = 100  # number of clusters (equivalent to the number of words) we want to estimate
    kmeans = KMeans(n_clusters=K, random_state=0, n_init=1)
    clusters = kmeans.fit(descriptors)  # we use the descriptors extracted from the map (training) images before
    centroids = clusters.cluster_centers_
    return clusters, centroids

def bag_of_words(centroids, img_descriptors):  
    '''
    returns the bag of words of the image with respect to the centroids
    '''
    n_centroids = centroids.shape[0]  # number of centroids found with the KMeans clustering
    n_descriptors = img_descriptors.shape[0]  # number of descriptors extracted from the image
    
    # initialization of the bag of words (BoW) vector
    # Note that the BoW vector has length equal to the number of cluster centroids
    # The cluster centroids are indeed our visual words, and the BoW will be the histogram of these words found in the given image
    bow_vector = np.zeros(n_centroids)  
    
    for n in range(n_descriptors):
        similarities = [np.sqrt(np.sum(np.square(img_descriptors[n]-centroid))) for centroid in centroids]
        best = np.argmin(similarities)
        bow_vector[best] += 1
    return bow_vector

def bag_of_words_matrix(centroids, descriptors ,func=bag_of_words):
    return np.vstack ( (func(centroids, img_des) for img_des in descriptors) )

bag_of_words_matrix = make_cached(bag_of_words_matrix, 'bow.npy')