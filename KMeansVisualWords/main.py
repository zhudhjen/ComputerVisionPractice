import os
import shutil
import cv2
import numpy as np

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
img = cv2.imread("origin.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# do sift
sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)

# do k-means
k = 40
compactness, labels, centers = cv2.kmeans(des, k, None, (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
                                                         10000, 0.001), 20, cv2.KMEANS_RANDOM_CENTERS)
# extract and classify visual words
words = [[] for i in range(k)]
length = 80
images_per_line = 15

for index, point in enumerate(kp):
    if point.size > 2:
        word = subimage(gray, point.pt, point.angle, point.size * 2)
        uniform_word = cv2.resize(word, (length, length))
        words[labels[index]].append(uniform_word)

# clean directories
if os.path.exists("words"):
    shutil.rmtree("words")
os.makedirs("words")

# put similar words into one image
for index, word in enumerate(words):
    if len(word) == 0:
        continue
    for i in range(images_per_line - 1 - (len(word) - 1) % images_per_line):
        word.append(255 * np.ones((length, length), dtype=np.uint8))
    image = np.vstack([np.hstack(word[i:i+images_per_line]) for i in range(0, len(word), images_per_line)])
    cv2.imwrite("words/word" + str(index) + ".png", image)

# draw keypoints
img = cv2.drawKeypoints(gray, kp, gray)
cv2.imwrite('sift_keypoints.png', img)
