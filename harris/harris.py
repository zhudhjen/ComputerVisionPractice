from PIL import Image, ImageDraw
import numpy as np

# function to generate local gaussian function
def makeGaussian(size, fwhm=3, center=None):

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


# function to draw a cross on a image
def drawCross(im, x, y, color):
    r = 5
    draw = ImageDraw.Draw(im)
    draw.line((x - r, y - r, x + r, y + r), width = 2, fill=color)
    draw.line((x - r, y + r, x + r, y - r), width = 2, fill=color)


# function to draw a series of crosses on image
def drawCorners(im, corners=None, color="black", title=""):
    if not corners:
        corners = []
    im_show = im.copy()
    for point in corners:
        drawCross(im_show, point[1], point[0], color)

    draw = ImageDraw.Draw(im_show)
    draw.text((im_show.size[0] - 100, 2), title, "black")

    im_show.show()

    im_show.save(title + ".bmp")


# function to do non-maximum suppression on image
def nonmaxSuppress(response_matrix, threshold, suppress_radius):

    corners = []
    local_maximum = []

    height = len(response_matrix)
    width = len(response_matrix[0])

    # calculate local maximum matrix
    for i in range(height):
        line = []
        for j in range(width):
            # find local maximum
            range_x = range(max(i - suppress_radius, 0), min(i + suppress_radius + 1, height))
            range_y = range(max(j - suppress_radius, 0), min(j + suppress_radius + 1, width))
            max_value = 0
            for x in range_x:
                for y in range_y:
                    if response_matrix[x][y] > max_value:
                        max_value = response_matrix[x][y]
            line.append(max_value)
        local_maximum.append(line)

    # extract local maximums to corners list
    for i in range(height):
        for j in range(width):
            # use threshold to filter out blank space and lines
            if response_matrix[i][j] == local_maximum[i][j] and local_maximum[i][j] > threshold:
                corners.append((i, j))

    return corners


# main program

im = Image.open('complex_original.png')

drawCorners(im, title="Original image")

pixels = list(im.getdata())
width, height = im.size
pixels = [[sum(color) / 3 for color in pixels[i * width:(i + 1) * width]] for i in range(height)]

# calculate first derivative
dx = [[pixels[i + 1][j] - pixels[i][j] for j in range(width)] for i in range(height - 1)]
dy = [[pixels[i][j + 1] - pixels[i][j] for j in range(width - 1)] for i in range(height)]

# calculate second derivative
dxx = [[dx[i + 1][j] - dx[i][j] for j in range(width)] for i in range(height - 2)]
dyy = [[dy[i][j + 1] - dy[i][j] for j in range(width - 2)] for i in range(height)]
dxy = [[dx[i][j + 1] - dx[i][j] for j in range(width - 1)] for i in range(height - 1)]

# Hessian detector
hessian = [[abs(dxx[i][j] * dyy[i][j] - dxy[i][j] ** 2) for j in range(width - 2)] for i in range(height - 2)]

hessian_corners = nonmaxSuppress(hessian, threshold=10000, suppress_radius=6)
drawCorners(im, hessian_corners, "red", "Hessian detector")

# Harris detector
window_radius = 4
k = 0.05

gaussian = makeGaussian(2 * window_radius + 1)
square = np.ones((2 * window_radius + 1, 2 * window_radius + 1))

window_function = gaussian

harris = []
for i in range(height - 2):
    line = []
    for j in range(width - 2):
        # avoid going out of boundary
        range_x = range(max(i - window_radius, 0), min(i + window_radius + 1, height - 2))
        range_y = range(max(j - window_radius, 0), min(j + window_radius + 1, width - 2))

        # sum up total gaussian weight in window to calculate weighted mean
        weight_sum = 0

        sum_xx = 0
        sum_xy = 0
        sum_yy = 0
        for x in range_x:
            for y in range_y:
                window_weight = window_function[x - i][y - j]
                sum_xx += window_weight * dxx[x][y]
                sum_xy += window_weight * dxy[x][y]
                sum_yy += window_weight * dyy[x][y]
                weight_sum += window_weight

        # calculate local weighted mean
        sum_xx /= weight_sum
        sum_xy /= weight_sum
        sum_yy /= weight_sum

        # to avoid calculating the eigenvalues, convert the formula to use det and trace
        det = sum_xx * sum_yy - sum_xy ** 2
        trace = sum_xx + sum_yy

        line.append(abs(det - k * trace ** 2))
    harris.append(line)

harris_corners = nonmaxSuppress(harris, threshold=400, suppress_radius=window_radius * 2)
drawCorners(im, harris_corners, "blue", "Harris detector")

