import cv2
import numpy as np

from . import gridlines


def find_in_check(im, clf):
    # crop bottom right of image where signature lies, according to our prior knowledge
    print('\tlooking at bottom-right of the check...')
    h, w = im.shape
    im = im[h / 2:h, w / 2:w]

    # Thresholding to get binary image
    print('\tremoving background...')
    thresh = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 30)

    # Remove lines from the image
    print('\tremoving grid lines...')
    thresh = gridlines.remove(thresh, fill=True)

    # Extract connected components
    print('\tanalysing connected components to find signature...')
    connectivity = 8  # 4-way / 8-way connectivity
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)

    # Label extracted components
    print('\tthinking...')
    for label in range(count):
        # Crop out the component
        x, y, w, h = stats[label, 0], stats[label, 1], stats[label, 2], stats[label, 3]
        component = thresh[y: y + h,
                    x: x + w]

        # The indexes of pixels belonging to char are stored in image
        im = np.where(labels == label)

        # Extract SURF features from the component
        surf = cv2.xfeatures2d.SURF_create()
        surf.setHessianThreshold(400)  # using 400 Hessian threshold
        kp, des = surf.detectAndCompute(component, None)  # keypoints, descriptors

        if des is not None:
            # Classify each descriptor of the component (to build consensus)
            rows = des.shape[0]
            predictions = np.zeros(rows)
            for row in range(rows):
                predictions[row] = clf.predict(des[row].reshape(1, -1))

            # Component marked signature only if >99% sure
            votes_all = len(predictions)
            votes_yes = np.count_nonzero(predictions)
            confidence = 100.0 * votes_yes / votes_all
            if confidence < 1:
                thresh[im] = 0
        else:
            thresh[im] = 0

    # Invert colors
    thresh = cv2.bitwise_not(thresh)

    # Approximate bounding box around signature
    points = np.argwhere(thresh == 0)  # find where the black pixels are
    points = np.fliplr(points)  # store them in x,y coordinates instead of row,col indices
    x, y, w, h = cv2.boundingRect(points)  # create a rectangle around those points
    x, y, w, h = x - 10, y - 10, w + 20, h + 20  # add padding

    print("\tsignature location reported")
    return x, y, w, h


def find_in_grid(im_grid):
    # Threshold  to get binary image
    ret3, thresh = cv2.threshold(im_grid, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove grid lines from the image
    im_grid = gridlines.remove(thresh)

    # Divide 4x2 grid into two 2x2 grids
    h, w = im_grid.shape
    grid_1 = im_grid[0:h / 2, 0:w]
    grid_2 = im_grid[h / 2:h, 0:w]

    # Extract all eight signatures from the grid
    _h, _w = np.array(grid_1.shape) / 2
    py, px = int(_h * 0.05), int(_w * 0.05)

    _h, _w = _h - 2 * py, _w - 2 * px

    signatures = []
    for x in [px, _w]:
        for y in [py, _h]:
            signatures.append((grid_1[y:y + _h, x:x + _w], (x, y, _w, _h)))
            signatures.append((grid_2[y:y + _h, x:x + _w], (x, h / 2 + y, _w, _h)))

    boxes = []
    for signature, rel_boxes in signatures:
        # Invert colors
        signature = cv2.bitwise_not(signature)

        # Approximate bounding box around signature
        points = np.argwhere(signature == 0)  # find where the black pixels are
        points = np.fliplr(points)  # store them in x,y coordinates instead of row,col indices
        x, y, w, h = cv2.boundingRect(points)  # create a rectangle around those points
        x, y, w, h = x - 10, y - 10, w + 20, h + 20  # add padding
        _x, _y, _w, _h = rel_boxes
        boxes.append((x + _x, y + _y, w, h))

    return boxes


def extract_from_check(im, model):
    print("locating signature in check...")
    x, y, dx, dy = find_in_check(im, model)

    # Crop out the signature (we assume it's in bottom-right corner)
    h, w = im.shape
    x += w / 2
    y += h / 2
    return im[y:y + dy, x:x + dx]


def extract_from_grid(im_grid):
    boxes = find_in_grid(im_grid)

    signatures = []
    for x, y, dx, dy in boxes:
        signatures.append(im_grid[y:y + dy, x:x + dx])

    return signatures
