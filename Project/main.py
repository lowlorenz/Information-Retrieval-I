from ImageHandler import *
from Utils import *
from skimage.feature import ORB

handler = ImageHandler()
#plot([handler.query_images_gray[0]], cols=1)

descriptors = extract_ORB(handler)

print(descriptors.shape)