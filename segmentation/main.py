from skimage import io, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt
from mean_shift import mean_shift

image = io.imread("94079.jpg")
bandwidths = range(10, 30, 10)
ms_output = []
for b in bandwidths:
    new_image = mean_shift(image, b)
    ms_output.append(new_image)
    io.imsave("94079_ms_b" + b + ".jpg", new_image)

print("Computing N-Cut ...")
label_slic = segmentation.slic(image, compactness=20, n_segments=600)
mean = graph.rag_mean_color(image, label_slic, mode='similarity')
label_ncut = graph.cut_normalized(label_slic, mean)
ncut_output = color.label2rgb(label_ncut, image, kind='avg')

plt.figure().suptitle('Original')
io.imshow(image)

for b in bandwidths:
    plt.figure().suptitle('Result of Mean Shift - Bandwidth = ' + b)
    io.imshow(ms_output[b])

plt.figure().suptitle('Result of N-Cut')
io.imshow(ncut_output)
io.show()
