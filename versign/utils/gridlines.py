import cv2
import numpy as np


def erase_lines_x(im):
    lines = []

    h, w = im.shape
    columns = range(w)
    rows = range(h)

    for r in rows:  # For each horizontal scan line
        mean = np.mean(im[r, columns])  # calculate pixel density (i.e.
        pxd = mean / 255.0  # percentage of white pixels)

        if pxd > 0.25:  # For >25% white pixels, we make the
            lines.append(r)  # whole scan line (FIXME: Should be
            im[r, columns] = 0  # the actual line only) black.
            try:
                im[r - 2, columns] = 0
                im[r - 1, columns] = 0
                im[r + 1, columns] = 0
                im[r + 2, columns] = 0
            except:
                pass

    return im, lines


def erase_lines_y(im):
    lines = []

    h, w = im.shape
    columns = range(w)
    rows = range(h)

    for c in columns:  # For each vertical scan line
        mean = np.mean(im[rows, c])  # calculate pixel density (i.e.
        pxd = mean / 255.0  # percentage of white pixels)

        if pxd > 0.25:  # For >25% white pixels, we make the
            lines.append(c)  # whole scan line (FIXME: Should be
            im[rows, c] = 0  # the actual line only) black.
            try:
                im[rows, c - 2] = 0
                im[rows, c - 1] = 0
                im[rows, c + 1] = 0
                im[rows, c + 2] = 0
            except:
                pass
    return lines


def fill_x(im, x_lines):
    print("x-lines:", len(x_lines))
    h, w = im.shape
    for r in x_lines:
        for c in range(w):
            try:
                t_px = im[r + 1, c]
                c_px = im[r, c]
                b_px = im[r - 2, c]

                if c_px == 0 and t_px != 0 and b_px != 0:
                    for i in range(-5, 5):
                        im[r + i, c] = 255
            except:
                # Ignore index out of bounds errors
                pass

    return im


def fill_y(im, y_lines):
    print("y-lines:", len(y_lines))
    h, w = im.shape
    for c in y_lines:
        for r in range(h):
            try:
                r_px = im[r, c + 5]
                c_px = im[r, c]
                l_px = im[r, c - 5]

                if c_px == 0 and l_px != 0 and r_px != 0:
                    for i in range(-3, 4):
                        im[c + i, r] = 255
            except:
                # Ignore index out of bounds errors
                pass

    return im


def remove(image, kernel=(25, 25), fill=False):
    # Make a copy (original image will be needed later)
    copy = np.copy(image)

    # Remove all lines (horizontal and vertical)
    print('Removing horizontal lines')
    x_lines = erase_lines_x(copy)

    print('Removing vertical lines')
    y_lines = erase_lines_y(copy)

    # Remove noise (removes any parts of lines not removed)
    print('Removing residual noise')
    filter = cv2.GaussianBlur(copy, kernel, 0)
    ret3, copy = cv2.threshold(filter, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Fill in any holes left by line removal
    if fill is True:
        print('Filling holes')
        copy = fill_x(copy, x_lines)
        copy = fill_y(copy, y_lines)
    else:
        print('Filling disabled')

    # Gaussian filtering for noise removal thickens all strokes
    # and filling can sometimes color pixels which were unfilled
    # in original image. These side effects are reversed by
    # taking an intersection of the processed image with the
    # original image
    print('Un-filling false positives')
    return cv2.bitwise_and(copy, image)
