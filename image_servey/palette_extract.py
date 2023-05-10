import colorsys
import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt
import math
import cv2

# import torch
# import time

from sklearn.cluster import KMeans

# use_cuda = torch.cuda.is_available()
# dtype = torch.float32 if use_cuda else torch.float64
# device_id = "cuda:0" if use_cuda else "cpu"


def centroid_histogram(clt):
        # grab the number of different clusters and create a histogram
        # based on the number of pixels assigned to each cluster
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins = numLabels)
    
        # normalize the histogram, such that it sums to one
        hist = hist.astype("float")
        hist /= hist.sum()
    
        # return the histogram
        return hist
    
def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0
    colors = []
    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
            color.astype("uint8").tolist(), -1)
        colors.append(color.astype("uint8").tolist())
        startX = endX
    
    # return the bar chart
    return bar, colors

def plot_color(rgb, h, w):
    hsv = colorsys.rgb_to_hsv()
    rads = math.tau * hsv[0] - math.pi
    mag = hsv[1] * (h)

def sort_plot(img):
    img = np.asarray(img, dtype=np.uint8)
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    sorted_img = [[x.sum() / 3, x[0], x[1], x[2]] for x in img]
    print(sorted_img[:10])
    sorted_img = np.asarray(sorted(sorted_img, key=lambda x: x[0]))
    print(sorted_img[:10])

    plt.figure()
    plt.plot(sorted_img[:, 1], color='red', label="Red")
    plt.plot(sorted_img[:, 2], color='green', label="Green")
    plt.plot(sorted_img[:, 3], color='blue', label="Blue")

    plt.legend()
    plt.show()

def kmeans_palette(img, real=True, name="no_name"):
    img = np.asarray(img, dtype=np.uint8)
    

    # shape = img.shape
    # new_img = [[[0] * shape[2]] * shape[1]] * shape[0]
    img = img.reshape((img.shape[0] * img.shape[1], 3))

    clt = KMeans(n_clusters=7)
    clt.fit(img)

    hist = centroid_histogram(clt)
    bar, colors = plot_colors(hist, clt.cluster_centers_)
    print("K-means complete")

    plt.figure()
    plt.axis("off")
    plt.imshow(bar)
    # plt.show()

    # print(clt.labels_)
    c = [0] * len(clt.labels_)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    for i in range(len(clt.labels_)):
        c[i] = '#%02x%02x%02x' % (colors[clt.labels_[i]][0], colors[clt.labels_[i]][1], colors[clt.labels_[i]][2])
        # new_img[i] = [colors[clt.labels_[i]][0], colors[clt.labels_[i]][1], colors[clt.labels_[i]][2]]


    ax.scatter3D(img[:, 0], img[:, 1], img[:, 2], c=c[:], alpha=0.4, marker='o')
    ax.scatter3D(clt.cluster_centers_[:, 0], clt.cluster_centers_[:, 1], c='black')
    plt.show()
    # plt.close()
    if real:
        plt.savefig(f'./figures/real/{name}.png')
    else:
        plt.savefig(f'./figures/fake/{name}.png')
    print("Done: ", name)



def palette(img):
    arr = np.asarray(img)
    palette, index = np.unique(asvoid(arr).ravel(), return_inverse=True)
    palette = palette.view(arr.dtype).reshape(-1, arr.shape[-1])
    count = np.bincount(index)
    order = np.argsort(count)
    return palette[order[::-1]]

def asvoid(arr):
    arr = np.ascontiguousarray(arr)
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))

def write_palette(img):
    # arr = np.arange(0, 737280, 1, np.uint8)
    # arr = np.array((1000, 100, 3))
    arr = np.empty(shape=(100, 1500, 3), dtype=np.uint8)
    print(arr)

    for i in range(100):
        for ii in range(1000):
            for iii in range(3):
                # print(img[i//100][iii])
                arr[i][ii][iii] = img[ii//150][iii]

    # arr = np.asarray(arr, dtype=np.uint8)
    print(type(arr))
    print(arr.shape)
    data = Image.fromarray(arr, 'RGB')
    data.save('image-0.png')
    data.show()
    # plt.imshow(arr, interpolation='nearest')
    # plt.show()

# for i in range(11):
#     image = Image.open(f'static/fake_images/image-{i}.jpg', 'r').convert('RGB')
#     kmeans_palette(image, real=False, name=f'image-{i}')
# for i in range(11):
#     image = Image.open(f'static/real_images/image-{i}.jpg', 'r').convert('RGB')
#     kmeans_palette(image, real=True, name=f'image-{i}')

image = Image.open(f'static/real_images/image-15.jpg', 'r').convert('RGB')
# kmeans_palette(image, real=True, name=f'image-15')
sort_plot(image)
# print(palette(image)[:11])
# write_palette(palette(image)[:15])

