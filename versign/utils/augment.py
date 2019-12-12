import os
import random

from PIL import Image


class AutoAugment:
    def __random_rotate(self, image, a_range):
        # type: (Image, (float, float)) -> Image
        """
        Rotates an image around its center by a random angle in range.

        :param image: the image to rotate
        :param a_range: range of angles in degrees as (min, max)
        :return: rotated image
        """
        angle = random.uniform(a_range[0], a_range[1])
        return self.__rotate(image, angle)

    @staticmethod
    def __rotate(image, angle):
        # type: (Image, float) -> Image
        """
        Rotate an image around its center by an angle .

        :param image: the image to rotate
        :param angle: the angle in degrees
        :return: rotated image
        """
        rgb = image.convert("RGBA")
        rot = rgb.rotate(angle, expand=True)
        fff = Image.new("RGBA", rot.size, (255,) * 4)
        out = Image.composite(rot, fff, rot)
        return out.convert(image.mode)

    def augment(self, images, angle=5.0, copies=5):
        # type: ([Image], float, int) -> [Image]
        """
        Perform data augmentation on some images to generate more images.

        :param images: list of images to augment
        :param angle: maximum angle of rotation (default:5.0)
        :param copies: copies to generate from each image (default: 5)
        :return: list of augmented images
        """
        augmented_images = []
        for image in images:
            for i in range(copies):
                augmented_images.append(self.__random_rotate(image, (-angle, angle)))

        return augmented_images


def auto_augment(in_dir, out_dir):
    """
    Performs data augmentation to generate more signature samples.

    :param in_dir: folder where sample images are
    :param out_dir: folder where augmented images will be saved
    """
    # Open all the images in input directory
    valid_exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
    filenames = [i for i in sorted(os.listdir(in_dir)) if os.path.splitext(i)[1] in valid_exts]
    images = [Image.open(os.path.join(in_dir, i)) for i in filenames]

    # Perform augmentation
    copies = 5
    augmented = AutoAugment().augment(images, angle=5.0, copies=copies)

    # Save augmented images in output directory
    for i, image in enumerate(augmented):
        file = filenames[int(i / copies)]
        name = os.path.splitext(file)[0]
        ext = os.path.splitext(file)[1].lower()

        outfile = os.path.join(out_dir, name + str(int(i % copies)) + ext)
        image.save(outfile)
