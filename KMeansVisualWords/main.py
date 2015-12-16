from os import listdir
from os.path import isfile, join
import shutil
import cv2
import numpy as np
from numpy.linalg import norm

# get tilted subimage
def subimage(image, center, theta, radius):
    theta *= np.pi / 180 # convert to rad

    v_x = (np.cos(theta), np.sin(theta))
    v_y = (-np.sin(theta), np.cos(theta))
    s_x = center[0] - (v_x[0] + v_y[0]) * (radius / 2)
    s_y = center[1] - (v_x[1] + v_y[1]) * (radius / 2)

    mapping = np.array([[v_x[0], v_y[0], s_x],
                        [v_x[1], v_y[1], s_y]])

    return cv2.warpAffine(image, mapping, (int(radius), int(radius)),
                          flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)


# main
origin_dir = "./learn"
learn_images = [cv2.imread(join(origin_dir, f)) for f in listdir(origin_dir)]
learn_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in learn_images]

# do sift
sift = cv2.xfeatures2d.SIFT_create()
learn_sift = []
for gray in learn_gray:
    kp, dst_sift = sift.detectAndCompute(gray, None)
    learn_sift.extend(dst_sift)

learn_sift = np.array(learn_sift)

# do k-means
print("K-Means Undergo...")
k = 1000
compactness, labels, centers = cv2.kmeans(learn_sift, k, None, (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
                                                                30, 1), 5, cv2.KMEANS_RANDOM_CENTERS)

print("done")
# extract and classify visual words
dict = np.zeros(k)

for i in range(len(learn_sift)):
    dict[labels[i]] += 1

dst_img = cv2.imread("dst.png")
dst_gray = cv2.cvtColor(dst_img, cv2.COLOR_BGR2GRAY)

kp, dst_sift = sift.detectAndCompute(dst_gray, None)

dst_words = np.zeros(k)

for point in dst_sift:
    point = np.array(point)
    nearest_center = np.array(centers[0])
    nearest_center_index = 0
    for i in range(k):
        center = np.array(centers[i])
        if norm(point - center) < norm(point - nearest_center):
            nearest_center_index = i
            nearest_center = center
    dst_words[nearest_center_index] += 1

similarity = np.dot(dst_words, dict) / dst_sift.size

print("Similarity:", similarity)
