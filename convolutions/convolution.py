from scipy import signal as sg
import numpy as np
from PIL import Image as img


def image2array(image_file):
    return np.array(img.open(image_file), dtype=np.float64)


def arr2img(arr, image_file):
    img.fromarray(arr.round().astype(np.uint8)).save(image_file)


def normalize(arr):
    # takes the abs of each element and divides it by the max and mults by 255
    # ensures that each value in the array is in between 0 and 255
    return 255. * np.absolute(arr)/np.max(arr)


def edges_convolution(image_matrix, vertical=True):
    img_mean = []
    if len(image_matrix.shape) > 2:
        # collapse RGB channels into a single channel if need be
        img_mean = image_matrix.mean(axis=-1)
        arr2img(img_mean, "royce2_color_avg.jpg")
        # use the correct one of horizontal or vertical kernel
    kernel = np.array([[1, -1]]) if vertical else np.array([[1], [-1]])
    # perform a convolution on the image
    convolved = sg.convolve(img_mean if len(image_matrix) > 2
                            else image_matrix, kernel, "valid")
    # ensures each value is in between 0 and 255 before writing the image
    normed = normalize(convolved)
    title = "royce2_conv_" + ("vertical" if vertical else "horizontal") + ".jpg"
    arr2img(normed, title)
    return img_mean if len(image_matrix.shape) > 2 else image_matrix


if __name__ == '__main__':
    image_matrix = image2array("royce-large.jpg")
    mean_matrix = edges_convolution(image_matrix)
    edges_convolution(image_matrix, False)  # for the horizontal convolution
    # convolution that sharpens an image
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = sg.convolve(mean_matrix, sharpen_kernel, "valid")
    sharp_normed = normalize(sharpened)
    arr2img(sharp_normed, "royce2_looking_sharp.jpg")
