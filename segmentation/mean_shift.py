from skimage import color
from multiprocessing import Pool
from functools import partial
import numpy as np
import sys, time


def _kernel_func(r, center, c=5):
    return np.exp(- np.sum(np.square(r - center), axis=1) / np.square(c) / 2)


def _do_mean_shift(pos, image, radius, bandwidth, eps):
    x_start = max(pos[0] - radius, 0)
    y_start = max(pos[1] - radius, 0)
    x_end = min(pos[0] + radius + 1, image.shape[0])
    y_end = min(pos[1] + radius + 1, image.shape[1])
    data = np.copy(image[x_start:x_end, y_start:y_end, :]).reshape(-1, 3)
    return _iterate_mean_shift(image[tuple(pos)], data, bandwidth, eps)


def _iterate_mean_shift(color, space, bandwidth, eps):
    weight = _kernel_func(space, color, bandwidth)
    new_color = np.average(space, axis=0, weights=weight)
    if np.linalg.norm(new_color - color) > eps:
        return _iterate_mean_shift(new_color, space, bandwidth, eps)
    else:
        return color


def _color_space_compression(segmentation, bandwidth):
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


def _floodfill_compression(image, bandwidth):
    color_map = np.zeros(image.shape[:2])
    color_table = {}
    current_color = 0
    for pos, tag in np.ndenumerate(color_map):
        if color_map[pos] == 0:
            current_color += 1
            average_color = _do_floodfill(pos, image, bandwidth, current_color, color_map)
            color_table[current_color] = average_color
    for pos, tag in np.ndenumerate(color_map):
        image[pos] = color_table[tag]


deltas = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])


def _do_floodfill(pos, image, bandwidth, current_color, color_map):
    n = 1
    sum = image[pos].copy()
    color_map[pos] = current_color
    points = [pos]
    while len(points) != 0:
        current_pos = points.pop(0)
        for delta in deltas:
            new_pos = tuple(np.array(current_pos) + delta)
            if 0 <= new_pos[0] < image.shape[0] and 0 <= new_pos[1] < image.shape[1]:
                if color_map[new_pos] == 0 and np.linalg.norm(image[new_pos] - image[current_pos]) < bandwidth:
                    n += 1
                    color_map[new_pos] = current_color
                    sum += image[new_pos].copy()
                    points.append(tuple(list(new_pos)))
    return sum / n


def mean_shift(image_rgb, radius=20, bandwidth=4, eps=1, proc_count=4):
    start = time.time()

    print("Mean Shift: Radius =", radius, ", Bandwidth =", bandwidth, ", EPS =", eps)
    print("Initializing ...")
    image = color.rgb2luv(image_rgb)
    coords = np.rollaxis(np.indices(image.shape[:2]), 0, 3).reshape(-1, 2)

    print("Iterating Points ...")
    partial_iterate = partial(_do_mean_shift, image=image, radius=radius, bandwidth=bandwidth, eps=eps)

    pool = Pool(proc_count)
    segmentation = np.array(pool.map(partial_iterate, coords))
    pool.close()
    pool.join()

    print("Post Processing ...")
    sys.setrecursionlimit(image.shape[0] * image.shape[1] + 100)
    segmentation = segmentation.reshape(image.shape)
    _floodfill_compression(segmentation, bandwidth)

    print("Ending ...")
    segmentation = color.luv2rgb(segmentation)

    end = time.time()
    print("Time elapsed:", end - start, "s")

    return segmentation
