
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from skimage.feature import ORB
from sklearn.mixture import GaussianMixture
import cv2
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

def extract_SIFT(img, sift):
    keypoints, descriptors = sift.detectAndCompute(img,None)
    return descriptors

def extract_map_SIFT(handler, sift):
    # otherwise extract them and safe them in a npy fil
    return np.vstack( (extract_SIFT(img, sift) for img in handler.map_images_gray) )

def extract_query_SIFT(handler, sift):
    return np.vstack( (extract_SIFT(img, sift) for img in handler.query_images_gray) )

def cluster_k_means(descriptors, K):
    # clustering
    kmeans = KMeans(n_clusters=K, random_state=0, n_init=1)
    clusters = kmeans.fit(descriptors)  # we use the descriptors extracted from the map (training) images before
    centroids = clusters.cluster_centers_
    return clusters, centroids

def gaussian_mixture_model(descriptors, n_components):
    # clustering
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.fit(descriptors)
    return gmm
    
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

def bag_of_words_gmm(gmm, img_descriptors):  
    '''
    returns the bag of words of the image with respect to the fitted GMM components
    '''
    n_components = gmm.n_components  # number of centroids found with the KMeans clustering
    
    # initialization of the bag of words (BoW) vector with length equal to the number of clusters
    bow_vector = np.zeros(n_components)  
    
    for desc in img_descriptors:
        posterior_probs = gmm.predict_proba(desc.reshape(1,-1))
        bow_vector = bow_vector + posterior_probs

    return bow_vector

def bag_of_words_gmm_all(gmm, descriptors, n_images, n_features):  
    '''
    returns the bag of words of the image with respect to the fitted GMM components
    '''
    bow = []
    for i in range(0, n_images*n_features, n_features):
        img_descriptors = descriptors[i:i+n_features]
        bow.append(bag_of_words_gmm(gmm, img_descriptors))

    return np.vstack(bow)

def bag_of_words_matrix(centroids, descriptors ,func=bag_of_words):
    return np.vstack ( (func(centroids, img_des) for img_des in descriptors) )

bag_of_words_matrix = make_cached(bag_of_words_matrix, 'bow.npy')

def retrieve_images(map_bow_vectors, query_bow, n_images):
        n_map_bow_vectors = map_bow_vectors.shape[0]
        most_similar = None  
        distances = np.array([np.sum(np.square(query_bow-map_bow_vectors[n])) for n in range(n_map_bow_vectors)])
        most_similar = np.argsort(distances)
        return most_similar[:n_images]