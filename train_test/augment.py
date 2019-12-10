#########################################################################################
# This script takes two arguments, inDir and outDir, and augments the input data in     #
# inDir by generating four new images for each image in the input data. Augmentation is #
# performed by performing minor transformations (rotation, etc.) on input data.         #
#########################################################################################
import os
import random
import sys

import cv2
import numpy as np
from PIL import Image

from preprocess import preprocess_signature, center_inside


def random_rotate(im, min_r, max_r):
    im2 = im.convert("RGBA")
    rot = im2.rotate(random.uniform(min_r, max_r), expand=True)
    fff = Image.new("RGBA", rot.size, (255,) * 4)
    out = Image.composite(rot, fff, rot)
    return out.convert(im.mode)


def augment(in_dir, out_dir, copies=4, max_a=5):
    """
    Performs data augmentation to generate more signature samples.

    :param in_dir: folder where sample images are
    :param out_dir: folder where augmented images will be saved
    :param copies: number of copies to create from each sample (default: 4)
    :param max_a: max angle to rotate through
    """
    valid_exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
    for file in os.listdir(in_dir):
        fn = os.path.splitext(file)[0]
        ext = os.path.splitext(file)[1].lower()
        if ext not in valid_exts:
            continue

        # Preprocess input image
        im = cv2.imread(os.path.join(in_dir, file), 0)
        im_proc = preprocess_signature(im, canvas_size=(150, 220))

        outfile = os.path.join(out_dir, fn + "0" + ext)
        cv2.imwrite(outfile, im_proc)

        # Augment input data by rotating image at random angles in range (-max_a, max_a)
        im = Image.open(outfile)
        for i in range(0, copies):
            rotated = random_rotate(im, min_r=-max_a, max_r=max_a)
            rotated = Image.fromarray(center_inside(np.array(rotated), canvas_size=(150, 220)))
            rotated.save(os.path.join(out_dir, fn + str(i + 1) + ext))


def main():
    # Validate command-line arguments
    if len(sys.argv) < 3:
        print("\nUsage:\tpython", sys.argv[0], "<input-folder> <output-folder>\n")
        return

    augment(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
