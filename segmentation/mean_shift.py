from skimage import color
from scipy.spatial import cKDTree
from multiprocessing import Pool
from functools import partial
import numpy as np
import sys, time


def _iterate_shift(pos, pixels, kd, bandwidth):
    neighbour_index = kd.query_ball_point(pos, bandwidth, 2)
    neighbours = pixels.take(neighbour_index, axis=0)
    new_pos = neighbours.mean(axis=0)
    if np.linalg.norm(new_pos - pos) > bandwidth / 10:
        return _iterate_shift(new_pos, pixels, kd, bandwidth)
    else:
        return pos


def mean_shift(image_rgb, bandwidth=10):
    start = time.time()

    print("Mean Shift: Bandwidth =", bandwidth, ", EPS =", bandwidth / 10)
    print("Initializing ...")
    image = color.rgb2luv(image_rgb)
    pixels = image.reshape(-1, 3)

    print("Constructing KD-Tree ...")
    sys.setrecursionlimit(10000)
    kd = cKDTree(pixels, 10)

    print("Iterating Points ...")
    partial_iterate = partial(_iterate_shift, pixels=pixels, kd=kd, bandwidth=bandwidth)

    pool = Pool(4)
    segmentation = np.array(pool.map(partial_iterate, pixels))

    print("Post Processing ...")
    center_frequency = {}
    for pixel in segmentation:
        pixel_tuple = tuple(pixel.tolist())
        if pixel_tuple not in center_frequency:
            center_frequency[pixel_tuple] = 0
        center_frequency[pixel_tuple] += 1

    sorted_center_frequency = np.array(sorted(center_frequency.items(), key=lambda tup: tup[1], reverse=True))
    color_map = {}
    centers = np.ndarray((0, 3))
    for center, frequency in sorted_center_frequency:
        distance = np.linalg.norm(centers - center, axis=1)
        matches = np.where(distance < bandwidth)
        if matches[0].size > 0:
            color_map[center] = centers[matches[0][0]]
        else:
            centers = np.vstack((centers, center))
            color_map[center] = center

    for index, pixel in enumerate(segmentation):
        segmentation[index] = color_map[tuple(pixel.tolist())]

    print("Ending ...")
    segmentation = segmentation.reshape(image.shape)
    segmentation = color.luv2rgb(segmentation)

    end = time.time()
    print("Time elapsed:", end - start, "s")

    return segmentation
