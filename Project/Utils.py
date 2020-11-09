
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from skimage.feature import ORB
import os


def extract_ORB(handler):

    descriptors = None

    # if descriptors are saved access them
    if os.path.isfile('orb.npy'):
        with open('orb.npy', 'rb') as f:
            descriptors = np.load(f)
            return descriptors

    # otherwise extract them and safe them in a npy file
    orb = ORB(n_keypoints=200)

    for img in handler.map_images_gray:
        # extract ORB 
        orb.detect_and_extract(img) 
        if descriptors is None:
            descriptors = orb.descriptors 
        else:
            descriptors = np.vstack((descriptors, orb.descriptors))
    
    with open('orb.npy', 'wb') as f:
        np.save(f, descriptors)
    
    return descriptors

        


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
        similarities = [np.sum(np.square(img_descriptors[n]-centroid)) for centroid in centroids]
        best = np.argmin(similarities)
        bow_vector[best] += 1
    return bow_vector