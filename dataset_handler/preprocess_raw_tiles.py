import os
import cv2

class PreprocessRawTiles:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def filter_images(self):
        os.makedirs(self.output_dir, exist_ok=True)

        for filename in os.listdir(self.input_dir):
            if filename.endswith('.png'):
                img_path = os.path.join(self.input_dir, filename)
                img = cv2.imread(img_path)

                height, width, _ = img.shape

                if height >= 150 and width >= 150:
                    output_path = os.path.join(self.output_dir, filename)
                    cv2.imwrite(output_path, img)

    def resize_images(self):
        for filename in os.listdir(self.output_dir):
            if filename.endswith('.png'):
                img_path = os.path.join(self.output_dir, filename)
                img = cv2.imread(img_path)

                resized_img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)

                output_path = os.path.join(self.output_dir, filename)
                cv2.imwrite(output_path, resized_img)

    def apply(self):
        self.filter_images()
        self.resize_images()
