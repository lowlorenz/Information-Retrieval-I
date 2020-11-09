from matplotlib import pyplot as plt
import numpy as np
import json
import h5py 


class ImageHandler:

    def __init__(self, path='data02/'):

        self.query_images = read_images(path, 'query.json')
        self.map_images = read_images(path, 'map.json')

        with h5py.File(path + 'gt.h5','r') as f:
            self.similar = f['sim'][:].astype(np.uint8)
        
    def similarity(self, query_index, map_index):
        return self.similar[query_index, map_index]


def read_images(path, file):
    with open(path + file,'r') as file:
        map_info = json.load(file)
        return [
            plt.imread(path + image_name) 
            for image_name in map_info['im_paths']
        ]

def plot(images, cols=5, size=(20,10)):
    plt.figure(figsize=size)
    for i, image in enumerate(images):
        plt.subplot(len(images) / cols + 1, cols, i + 1)
        plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    # create instance
    handler = ImageHandler()

    # plot images 
    plot([handler.map_images[0]], cols=1)

    # easy access to the similarities
    for i in range(len(handler.query_images)):
        print(handler.similarity(i, 0))

    # fast access also possible
    similars = np.argwhere(handler.similar[:,0] == 1)
    print(similars)