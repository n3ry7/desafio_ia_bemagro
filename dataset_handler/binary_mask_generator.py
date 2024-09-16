import os
from typing import Any, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm


class BinaryMaskGenerator:
    """
    A class to generate binary masks for plant segmentation.

    Attributes:
        input_folder (str): The path to the input folder containing the images.
        output_folder_masks (str): The path to the output folder for the generated masks.
    """

    def __init__(self, input_folder: str, output_folder_masks: str) -> None:
        self.input_folder = input_folder
        self.output_folder_masks = output_folder_masks

    def collect_pixels(self, sample_fraction: float = 0.1) -> np.ndarray:
        """
        Collect pixels from the input images and sample a subset of them.

        Args:
            sample_fraction (float, optional): The fraction of pixels to sample from each image.
            Defaults to 0.1.

        Returns:
            numpy.ndarray: A 2D array containing the sampled pixels.
        """
        all_pixels = []
        for filename in tqdm(os.listdir(self.input_folder), desc="Collecting pixels"):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(self.input_folder, filename)
                img = cv2.imread(file_path)
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                h, w = img_hsv.shape[:2]
                mask = np.random.choice(
                    [0, 1], size=(h, w), p=[1 - sample_fraction, sample_fraction]
                )
                sampled_pixels = img_hsv[mask.astype(bool)]

                all_pixels.append(sampled_pixels)

        return np.vstack(all_pixels)

    def create_global_clusters(self, pixels: np.ndarray, n_clusters: int = 3) -> KMeans:
        """
        Create global clusters using the collected pixels.

        Args:
            pixels (numpy.ndarray): A 2D array of pixels.
            n_clusters (int, optional): The number of clusters to create. Defaults to 3.

        Returns:
            sklearn.cluster.KMeans: The trained KMeans model.
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(pixels)
        return kmeans

    def identify_clusters(self, centers: np.ndarray) -> tuple[Any, int, int]:
        """
        Identify the plant, ground, and undefined clusters based on the cluster centers.

        Args:
            centers (numpy.ndarray): A 2D array of cluster centers.

        Returns:
            Tuple[int, int, int]: The indices of the plant, ground, and undefined clusters.
        """
        saturation = centers[:, 1]
        hue = centers[:, 0]

        plant_cluster = np.argmax(saturation)
        if 30 <= hue[plant_cluster] <= 90:
            ground_candidates = [i for i in range(3) if i != plant_cluster]
            ground_cluster = ground_candidates[np.argmin(saturation[ground_candidates])]
            undefined_cluster = [i for i in range(3) if i not in [plant_cluster, ground_cluster]][0]
        else:
            plant_cluster = np.argmax((30 <= hue) & (hue <= 90))
            ground_candidates = [i for i in range(3) if i != plant_cluster]
            ground_cluster = ground_candidates[np.argmin(saturation[ground_candidates])]
            undefined_cluster = [i for i in range(3) if i not in [plant_cluster, ground_cluster]][0]

        return plant_cluster, ground_cluster, undefined_cluster

    def create_masks(
        self, image_path: str, kmeans_model: KMeans
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create plant, ground, and undefined masks for a given image.

        Args:
            image_path (str): The path to the input image.
            kmeans_model (sklearn.cluster.KMeans): The trained KMeans model.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]:
                - The plant mask
                - The ground mask
                - The undefined mask
                - The original image
        """
        img = cv2.imread(image_path)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        pixels = img_hsv.reshape((-1, 3))

        labels = kmeans_model.predict(pixels)
        centers = kmeans_model.cluster_centers_

        plant_cluster, ground_cluster, undefined_cluster = self.identify_clusters(centers)

        plant_mask = (labels == plant_cluster).reshape(img.shape[:2])
        ground_mask = (labels == ground_cluster).reshape(img.shape[:2])
        undefined_mask = (labels == undefined_cluster).reshape(img.shape[:2])

        kernel = np.ones((3, 3), np.uint8)
        plant_mask = cv2.dilate(plant_mask.astype(np.uint8) * 255, kernel, iterations=3)
        plant_mask = cv2.erode(plant_mask, kernel, iterations=2)

        plant_mask = plant_mask.astype(np.uint8)
        ground_mask = ground_mask.astype(np.uint8)
        undefined_mask = undefined_mask.astype(np.uint8)

        return plant_mask, ground_mask, undefined_mask, img

    def apply(self) -> None:
        """
        Apply the binary mask generation process to the input folder and save the results.
        """
        if not os.path.exists(self.output_folder_masks):
            os.makedirs(self.output_folder_masks)

        all_pixels = self.collect_pixels()

        kmeans_model = self.create_global_clusters(all_pixels)

        for filename in tqdm(os.listdir(self.input_folder), desc="Processing images"):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(self.input_folder, filename)
                plant_mask, ground_mask, undefined_mask, original_img = self.create_masks(
                    file_path, kmeans_model
                )

                cv2.imwrite(
                    os.path.join(
                        self.output_folder_masks, f"{os.path.splitext(filename)[0]}_plant_mask.png"
                    ),
                    plant_mask,
                )

                masked_img = cv2.bitwise_and(original_img, original_img, mask=plant_mask)
