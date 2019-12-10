import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.misc import imresize
from skimage.filters import threshold_local


def resize(image, canvas_size, interpolation="bilinear"):
    # type: (np.array, tuple, str) -> np.array
    """
    Resize an image to specified size while preserving the aspect ratio.

    The longer side of the image is made equal to the output length of that
    side, and the other side is scaled in accordance with the aspect ratio.

    :param image: the image to be resized
    :param canvas_size: maximum size of the output image in pixels
    :param interpolation: type of interpolation to use (default 'bilinear')
    """
    ih, iw = image.shape
    aspect = iw / ih

    out_w, out_h = canvas_size
    if ih >= iw:
        out_w = int(out_h * aspect)
    else:
        out_h = int(out_w / aspect)

    return imresize(image, (out_h, out_w), interp=interpolation)


def center_inside(im, canvas_size):
    # type: (np.array, tuple) -> np.array
    """
    Centers an image inside a canvas.

    Image is resized to fit on a white canvas while preserving the aspect ratio.

    :param im: the image to be resized
    :param canvas_size: size of the canvas on which image is to be centered
    """
    out_w, out_h = canvas_size
    canvas = np.ones(shape=(out_h, out_w)).astype('uint8') * 255

    im = resize(im, canvas_size)
    ih, iw = im.shape

    pad_x = (out_w - iw) / 2
    pad_y = (out_h - ih) / 2

    canvas[pad_y:pad_y + ih, pad_x:pad_x + iw] = im
    return canvas


def threshold_and_crop(im):
    # type: (np.array) -> np.array
    # Apply a gaussian filter on the image to remove small components
    # Note: this is only used to define the limits to crop the image
    blur_radius = 2
    blurred_image = ndimage.gaussian_filter(im, blur_radius)

    # Binarize the image using OTSU's algorithm. This is used to find the center
    # of mass of the image, and find the threshold to remove background noise
    threshold, binarized_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Crop the image with a tight box
    r, c = np.where(binarized_image == 0)
    cropped = im[r.min(): r.max(), c.min(): c.max()]

    # Remove noise - anything higher than the threshold. Note that the image is still grayscale
    cropped[cropped > threshold] = 255
    return cropped


def preprocess_signature(im, canvas_size):
    im = threshold_and_crop(im)
    im = center_inside(im, canvas_size)
    return im


def preprocess_cheque(infile, outfile):
    # Open image in grayscale mode
    image = np.array(Image.open(infile).convert("L"))

    # Apply local OTSU thresholding
    block_size = 25
    adaptive_thresh = threshold_local(image, block_size, offset=15)
    binarized = image > adaptive_thresh
    binarized = binarized.astype(float) * 255
    binarized = Image.fromarray(binarized).convert("L")

    # Save binarized file
    binarized.save(outfile)
