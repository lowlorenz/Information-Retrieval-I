
import numpy as np

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