
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from skimage.feature import ORB
from sklearn.mixture import GaussianMixture
import cv2
from scipy.stats import multivariate_normal
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
    descriptors = None
    for img in handler.map_images_gray:
        orb.detect_and_extract(img)  
        descriptors_img = orb.descriptors  # descriptors (the feature vectors)
        # Accumulate the computed descriptors
        if descriptors is None:
            descriptors = descriptors_img
        else:
            descriptors = np.vstack( (descriptors, descriptors_img))
    return descriptors
    
extract_map_ORB_c = make_cached(extract_map_ORB, 'map_orb.npy')

def extract_query_ORB(handler):
    descriptors = None
    for img in handler.query_images_gray:
        orb.detect_and_extract(img)  
        descriptors_img = orb.descriptors  # descriptors (the feature vectors)
        # Accumulate the computed descriptors
        if descriptors is None:
            descriptors = descriptors_img
        else:
            descriptors = np.vstack( (descriptors, descriptors_img))
    return descriptors
    
extract_query_ORB_c = make_cached(extract_query_ORB, 'query_orb.npy')

def extract_SIFT(img, sift):
    keypoints, descriptors = sift.detectAndCompute(img,None)
    return descriptors

def extract_map_SIFT(handler, sift):
    # otherwise extract them and safe them in a npy fil
    return np.vstack( (extract_SIFT(img, sift) for img in handler.map_images_gray) )

extract_map_SIFT_c = make_cached(extract_map_SIFT, 'map_sift.npy')

def extract_query_SIFT(handler, sift):
    return np.vstack( (extract_SIFT(img, sift) for img in handler.query_images_gray) )

extract_query_SIFT_c = make_cached(extract_query_SIFT, 'map_sift.npy')

def cluster_k_means(descriptors, K):
    # clustering
    kmeans = KMeans(n_clusters=K, random_state=0, n_init=1)
    clusters = kmeans.fit(descriptors)  # we use the descriptors extracted from the map (training) images before
    centroids = clusters.cluster_centers_
    return clusters, centroids

def euclidian(descriptor, centroid):
    return np.sqrt(np.sum(np.square(descriptor-centroid)))

def manhatten(descriptor, centroid):
    return np.sum(np.abs(descriptor-centroid))

def gaussian_mixture_model(descriptors, n_components):
    # clustering
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.fit(descriptors)
    return gmm
    
def bag_of_words(centroids, img_descriptors, distance=euclidian):
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
        similarities = [ distance(img_descriptors[n],centroid) for centroid in centroids]
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

bag_of_words_gmm_all_c = make_cached(bag_of_words_gmm_all, 'bow_gmm.npy')

def bag_of_words_sift_all(centroids, descriptors, n_images, n_features):  
    '''
    returns the bag of words of the image with respect to the fitted GMM components
    '''
    bow = []
    for i in range(0, n_images*n_features, n_features):
        img_descriptors = descriptors[i:i+n_features]
        bow.append(bag_of_words(centroids, img_descriptors))

    return np.vstack(bow)

bag_of_words_sift_all_c = make_cached(bag_of_words_sift_all, 'bow_sift.npy')

def bag_of_words_gmm_unnormalised(gmm, img_descriptors):  
    '''
    returns the bag of words of the image with respect to the fitted GMM components
    '''
    n_components = gmm.n_components  # number of centroids found with the KMeans clustering
    
    # initialization of the bag of words (BoW) vector with length equal to the number of clusters
    bow_vector = np.zeros(n_components)  
    
    # Obtain the Gaussian distributions for each cluster/component. We'll use to find the likelihood.
    gaussians = [ multivariate_normal(gmm.means_[i], gmm.covariances_[i], allow_singular=True) 
                     for i in range(gmm.n_components)]
    log_prior = np.log(gmm.weights_)
    
    for desc in img_descriptors:
        log_probs = [C.logpdf(desc) + log_prior[i] for i,C in enumerate(gaussians)]
        unnormalised_posteriors = np.exp(log_probs)
        
        bow_vector = bow_vector + unnormalised_posteriors

    return bow_vector

def bag_of_words_gmm_unnormalised_all(gmm, descriptors, n_images, n_features):  
    '''
    returns the bag of words of the image with respect to the fitted GMM components
    '''
    bow = []
    for i in range(0, n_images*n_features, n_features):
        print(i)
        img_descriptors = descriptors[i:i+n_features]
        bow.append(bag_of_words_gmm_unnormalised(gmm, img_descriptors))

    return np.vstack(bow)

bag_of_words_gmm_unnormalised_all_c = make_cached(bag_of_words_gmm_unnormalised_all, 'bow_gmm_un.npy')


def bag_of_words_matrix(centroids, images, distance=euclidian):
    bow_images = None
    # loop over the images in the map set
    for i,img in enumerate(images):

        print(f'{i}/{len(images)}')
        # extract the keypoints and corresponding descriptors (200 ORB descriptors)
        orb.detect_and_extract(img)
        img_descriptors = orb.descriptors  # descriptors (the feature vectors)
        
        # compute BoW representation of the image (using the basic 'words', i.e. centroinds, computed earlier)
        bow = bag_of_words(centroids, img_descriptors, distance=distance)
        # add the computed BoW vector to the set of map representations
        if bow_images is None:
            bow_images = bow
        else:
            bow_images = np.vstack( (bow_images, bow))
    return bow_images

bag_of_words_matrix_c_euclidian = make_cached(bag_of_words_matrix, 'bow_euc.npy')
bag_of_words_matrix_c_manhatten = make_cached(bag_of_words_matrix, 'bow_man.npy')

# receives as input the:
#   - bag of words vectors of the map images
#   - the bag of work vector of the query image
def retrieve_images(map_bow_vectors, query_bow, distance):
    n_map_bow_vectors = map_bow_vectors.shape[0]
    most_similar = None  # use this to
    distances = np.array([ 
        distance(query_bow, map_bow_vectors[n]) 
        for n in range(n_map_bow_vectors) 
        ])
    most_similar = np.argsort(distances)
    distances = np.sort(distances)
    return most_similar, distances