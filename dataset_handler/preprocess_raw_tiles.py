import os

import cv2


class PreprocessRawTiles:
    """
    A class to preprocess raw image tiles.

    Attributes:
        input_dir (str): The path to the input directory containing the raw image tiles.
        output_dir (str): The path to the output directory where the preprocessed
        images will be saved.
    """

    def __init__(self, input_dir: str, output_dir: str) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir

    def filter_images(self) -> None:
        """
        Filter the input images by size and save the valid ones to the output directory.

        Raises:
            OSError: If the output directory cannot be created.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        for filename in os.listdir(self.input_dir):
            if filename.endswith(".png"):
                img_path = os.path.join(self.input_dir, filename)
                img = cv2.imread(img_path)

                height, width, _ = img.shape

                if height >= 150 and width >= 150:
                    output_path = os.path.join(self.output_dir, filename)
                    cv2.imwrite(output_path, img)

    def resize_images(self) -> None:
        """
        Resize the images in the output directory to 256x256 pixels.
        """
        for filename in os.listdir(self.output_dir):
            if filename.endswith(".png"):
                img_path = os.path.join(self.output_dir, filename)
                img = cv2.imread(img_path)

                resized_img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)

                output_path = os.path.join(self.output_dir, filename)
                cv2.imwrite(output_path, resized_img)

    def apply(self) -> None:
        """
        Apply the preprocessing steps to the input directory.
        """
        self.filter_images()
        self.resize_images()
