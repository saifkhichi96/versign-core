from os import makedirs, walk
from os.path import join, splitext, exists

from PIL import Image

from . import preprocess_signature


def preprocess_signatures(in_dir, out_dir, canvas_size):
    """
    Performs preprocessing on all images in a folder (including sub-folders).

    Preprocessing includes thresholding the image and centering it on a canvas of fixed size.
    All images in given directory are centered inside a white canvas of specified size
    using the given interpolation method, and then saved at the given output location.
    Accepted image formats include PNG, JPG, and TIFF.

    :param in_dir: dir in which images to be resized are located
    :param out_dir: dir where resized images should be saved
    :param canvas_size: size of the canvas on which image is to be centered
    """
    valid_exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    for path, dirs, files in walk(in_dir):
        files = [f for f in files if splitext(f)[1].lower() in valid_exts]

        for i, f in enumerate(files):
            save_dir = join(out_dir, path[len(in_dir):])
            if not exists(save_dir):
                makedirs(save_dir)

            infile = join(path, f)
            outfile = join(save_dir, splitext(f)[0] + ".png")

            preprocess_signature(Image.open(infile).convert("L"), canvas_size).save(outfile)
