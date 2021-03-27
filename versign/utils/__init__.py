import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.filters import threshold_local


def resize(image, canvas_size):
    # type: (np.array, tuple) -> np.array
    """Resize an image to specified size while preserving the aspect ratio.

    The longer side of the image is made equal to the output length of that
    side, and the other side is scaled in accordance with the aspect ratio.

    :param image: the image to be resized
    :param canvas_size: maximum size of the output image in pixels
    """
    ih, iw = image.shape
    aspect = iw / ih

    out_w, out_h = canvas_size
    if out_w < out_h:
        out_h = int(out_w / aspect)
        if out_h > canvas_size[1]:
            out_h = canvas_size[1]
            out_w = int(out_h * aspect)
    else:
        out_w = int(out_h * aspect)
        if out_w > canvas_size[0]:
            out_w = canvas_size[0]
            out_h = int(out_w / aspect)

    return np.array(Image.fromarray(image).resize((out_w, out_h), Image.BICUBIC))


def center_inside(im, canvas_size):
    # type: (np.array, tuple) -> np.array
    """Centers an image inside a canvas.

    Image is resized to fit on a black canvas while preserving the aspect ratio.

    Parameters:
        im (np.array) : the image to be resized
        canvas_size (tuple): the size (w,h) of the canvas

    Returns:
        input image centred on a black canvas of given size
    """
    out_w, out_h = canvas_size
    canvas = np.zeros(shape=(out_h, out_w)).astype('uint8') * 255

    im = resize(im, canvas_size)
    ih, iw = im.shape

    pad_x = int((out_w - iw) / 2)
    pad_y = int((out_h - ih) / 2)

    canvas[pad_y:pad_y + ih, pad_x:pad_x + iw] = im
    return canvas


def threshold_and_crop(im):
    """Removes signature background and padding.

    The image is thresholded using the OTSU's algorithm, and the background
    pixels are set to white (intensity 255), leaving the foreground pixels
    in grayscale. The image is then inverted such that the background is
    zero-valued.

    Parameters:
        im (np.array) : the signature image array to be thresholded

    Returns:
        thresholded and cropped signature image
    """
    # Threshold using OTSU's method
    retval, thresh = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Crop the original image with a tight box around signature
    r, c = np.where(thresh != 0)
    cropped = thresh[r.min(): r.max(), c.min(): c.max()]
    return cropped


def preprocess_signature(im, canvas_size):
    """Preprocesses a signature image.

    Parameters:
        im (np.array) : the image to be preprocessed
        canvas_size (tuple) : the size (w,h) of the resize canvas

    Returns:
        the preprocessed image as an numpy array
    """
    im = threshold_and_crop(im)
    im = center_inside(im, canvas_size)
    return im
