from skimage import io, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt
from mean_shift import mean_shift

image_names = ["sample/campus", "108005/108005", "25098/25098", "78004/78004", "94079/94079"]

for image_name in image_names:
    print("Processing image:", image_name)
    image = io.imread(image_name + ".jpg")
    bandwidths = range(20, 101, 20)
    ms_output = {}
    for b in bandwidths:
        new_image = mean_shift(image, b)
        ms_output[b] = new_image
        io.imsave(image_name + "_ms_local_b" + str(b) + ".jpg", new_image)

    print("Computing N-Cut ...")
    label_slic = segmentation.slic(image, compactness=20, n_segments=600)
    mean = graph.rag_mean_color(image, label_slic, mode='similarity')
    label_ncut = graph.cut_normalized(label_slic, mean)
    ncut_output = color.label2rgb(label_ncut, image, kind='avg')
    io.imsave(image_name + "_ncut.jpg", ncut_output)
    print()

    # plt.figure().suptitle('Original')
    # io.imshow(image)
    #
    # for b in bandwidths:
    #     plt.figure().suptitle('Result of Mean Shift - Bandwidth = ' + str(b))
    #     io.imshow(ms_output[b])
    #
    # plt.figure().suptitle('Result of N-Cut')
    # io.imshow(ncut_output)
    # io.show()
