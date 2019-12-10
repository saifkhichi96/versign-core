import os
import sys

import cv2

from preprocess import preprocess_signature


def preprocess_signatures(in_dir, out_dir, canvas_size, interpolation="bilinear"):
    """
    Performs preprocessing on all images in a folder (including sub-folders).

    Preprocessing includes thresholding the image and centering it on a canvas of fixed size.
    All images in given directory are centered inside a white canvas of specified size
    using the given interpolation method, and then saved at the given output location.
    Accepted image formats include PNG, JPG, TIFF, and BMP.

    :param in_dir: dir in which images to be resized are located
    :param out_dir: dir where resized images should be saved
    :param canvas_size: size of the canvas on which image is to be centered
    :param interpolation: type of interpolation to use (default: 'bilinear')
    """
    if not in_dir.endswith("/"):
        in_dir += "/"

    if not out_dir.endswith("/"):
        out_dir += "/"

    print("preprocess_signatures('%s'):" % in_dir)
    valid_exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
    for path, dirs, files in os.walk(in_dir):
        files = [f for f in files if os.path.splitext(f)[1].lower() in valid_exts]
        print("\t%d images found" % len(files))
        for i, f in enumerate(files):
            print("\r\tprocessing %d/%d" % (i + 1, len(files)), end="")
            save_dir = os.path.join(out_dir, path[len(in_dir):])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            infile = os.path.join(path, f)
            outfile = os.path.join(save_dir, os.path.splitext(f)[0] + ".png")

            im = cv2.imread(infile, 0)
            preprocess_signature(im, canvas_size)
            cv2.imwrite(outfile, im)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Invalid number of arguments.\nUsage:\n\tpython %s <in_dir> <out_dir> <canvas_size>" % (
            os.path.basename(__file__)))

        exit(-1)

    preprocess_signatures(sys.argv[1], sys.argv[2], sys.argv[3])
