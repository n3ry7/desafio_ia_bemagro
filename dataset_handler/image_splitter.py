import cv2
import os
from tqdm import tqdm

class ImageSplitter:
    def __init__(self, input_path, output_dir, tile_size=(256, 256)):
        self.input_path = input_path
        self.output_dir = output_dir
        self.tile_size = tile_size

    def apply(self):
        # Open the large TIF image
        image = cv2.imread(self.input_path, cv2.IMREAD_UNCHANGED)

        # Get the dimensions of the image
        height, width, _ = image.shape

        # Crop the image to remove the black regions
        # Find the first and last non-black column
        start_col = 0
        end_col = width - 1
        while start_col < width and image[:, start_col].sum() == 0:
            start_col += 1
        while end_col >= 0 and image[:, end_col].sum() == 0:
            end_col -= 1

        # Find the first and last non-black row
        start_row = 0
        end_row = height - 1
        while start_row < height and image[start_row, :].sum() == 0:
            start_row += 1
        while end_row >= 0 and image[end_row, :].sum() == 0:
            end_row -= 1

        # Crop the image
        image = image[start_row:end_row+1, start_col:end_col+1]

        # Adjust the tile size to match the cropped image dimensions
        self.tile_size = (min(self.tile_size[0], image.shape[1]), min(self.tile_size[1], image.shape[0]))

        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Iterate over the image and save the tiles as PNG files
        total_tiles = (image.shape[1] // self.tile_size[0]) * (image.shape[0] // self.tile_size[1])
        with tqdm(total=total_tiles, unit='tile', desc='Splitting orthomosaic') as pbar:
            for x in range(0, image.shape[1], self.tile_size[0]):
                for y in range(0, image.shape[0], self.tile_size[1]):
                    tile = image[y:y+self.tile_size[1], x:x+self.tile_size[0]]
                    cv2.imwrite(f'{self.output_dir}/tile_{x}_{y}.png', tile)
                    pbar.update(1)
