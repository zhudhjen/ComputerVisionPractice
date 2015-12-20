from skimage import color
from scipy.spatial import cKDTree
from multiprocessing import Pool
from functools import partial
import numpy as np
import sys, time


def _kernel_func(r, center, c=50):
    return np.exp(- np.square(r - center) / c)


def _do_mean_shift(pos, image, bandwidth, eps):
    # print(pos)
    x_start = max(pos[0] - bandwidth, 0)
    y_start = max(pos[1] - bandwidth, 0)
    x_end = min(pos[0] + bandwidth + 1, image.shape[0])
    y_end = min(pos[1] + bandwidth + 1, image.shape[1])
    data = np.copy(image[x_start:x_end, y_start:y_end, :]).reshape(-1, 3)
    return _iterate_mean_shift(image[tuple(pos)], data, eps)


def _iterate_mean_shift(color, space, eps):
    # print(color)
    weight = _kernel_func(space, color)
    new_color = np.average(space, axis=0, weights=weight)
    if np.linalg.norm(new_color - color) > eps:
        return _iterate_mean_shift(new_color, space, eps)
    else:
        return color


def mean_shift(image_rgb, bandwidth=10, eps=1, proc_count=4):
    start = time.time()

    print("Mean Shift: Bandwidth =", bandwidth, ", EPS =", eps)
    print("Initializing ...")
    image = color.rgb2luv(image_rgb)
    coords = np.rollaxis(np.indices(image.shape[:2]), 0, 3).reshape(-1, 2)

    """
    print("Constructing KD-Tree ...")
    sys.setrecursionlimit(10000)
    kd = cKDTree(pixels, 10)
    """

    print("Iterating Points ...")
    partial_iterate = partial(_do_mean_shift, image=image, bandwidth=bandwidth, eps=eps)

    pool = Pool(4)
    segmentation = np.array(pool.map(partial_iterate, coords))

    """
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
    """

    print("Ending ...")
    segmentation = segmentation.reshape(image.shape)
    segmentation = color.luv2rgb(segmentation)

    end = time.time()
    print("Time elapsed:", end - start, "s")

    return segmentation
