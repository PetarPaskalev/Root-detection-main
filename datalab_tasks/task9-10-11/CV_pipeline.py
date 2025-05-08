import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from patchify import patchify, unpatchify
from skimage.morphology import remove_small_objects, skeletonize
from skimage.measure import label
from skan import Skeleton, summarize

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

class Pipeline:
    def __init__(
        self,
        model_path,
        patch_size=256,
        plate_size_mm=150,
        plate_origin_in_robot=(0.10775, 0.062 - 0.026, 0.057),  # example offset
        morph_min_size=200,
        morph_kernel_size=5,
        morph_dilate_iter=2,
        morph_close_iter=1
    ):
        """
        Initializes the pipeline, loads the U-Net model, and stores parameters.
        """
        self.patch_size = patch_size
        self.plate_size_mm = plate_size_mm
        self.plate_origin_in_robot = plate_origin_in_robot

        self.morph_min_size = morph_min_size
        self.morph_kernel_size = morph_kernel_size
        self.morph_dilate_iter = morph_dilate_iter
        self.morph_close_iter = morph_close_iter

        # We will store the top/left offsets from detect_edges to do your row->mm_x mapping
        self.crop_left = 0
        self.crop_top = 0

        # For unpadding
        self.extra_h = 0
        self.extra_w = 0

        # Custom F1 for your model (if needed)
        def f1(y_true, y_pred):
            def recall_m(y_true, y_pred):
                TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
                return TP / (Positives + K.epsilon())

            def precision_m(y_true, y_pred):
                TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
                return TP / (Pred_Positives + K.epsilon())

            precision = precision_m(y_true, y_pred)
            recall = recall_m(y_true, y_pred)
            return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[ERROR] U-Net model not found at: {model_path}")
        self.model = load_model(model_path, custom_objects={"f1": f1})
        print(f"[INFO] Pipeline initialized with model: {model_path}")

    ###########################################################################
    # Basic I/O
    ###########################################################################
    def load_grayscale_image(self, image_path):
        if not os.path.exists(image_path):
            print(f"[ERROR] Image path does not exist: {image_path}")
            return None
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[ERROR] Failed to load image from: {image_path}")
        return img

    ###########################################################################
    # Cropping & Padding
    ###########################################################################
    def detect_edges(self, gray_image, max_size=2800):
        """
        Return (left, right, top, bottom) bounding box of the largest contour (the dish).
        """
        blurred = cv2.GaussianBlur(gray_image, (51, 51), 0)
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = cv2.magnitude(sobel_x, sobel_y)

        _, edges = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
        edges = edges.astype(np.uint8)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return (0, gray_image.shape[1], 0, gray_image.shape[0])

        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)

        side_length = min(max(w, h), max_size)
        cx, cy = x + w // 2, y + h // 2
        half = side_length // 2

        new_x = max(cx - half, 0)
        new_y = max(cy - half, 0)
        new_w = new_h = min(
            side_length,
            min(gray_image.shape[1] - new_x, gray_image.shape[0] - new_y)
        )

        left = new_x
        right = new_x + new_w
        top = new_y
        bottom = new_y + new_h
        return (left, right, top, bottom)

    def crop_image(self, image, edges):
        """
        Crop based on (left, right, top, bottom).
        We'll store these offsets in self, so we can do row->mm_x later.
        """
        left, right, top, bottom = edges
        self.crop_left = left
        self.crop_top = top
        return image[top:bottom, left:right]

    def pad_image(self, image):
        """
        Pads the image so (H%patch_size==0) and (W%patch_size==0).
        We'll store self.extra_h, self.extra_w to unpad later.
        """
        h, w = image.shape
        self.extra_h = (self.patch_size - (h % self.patch_size)) if (h % self.patch_size) else 0
        self.extra_w = (self.patch_size - (w % self.patch_size)) if (w % self.patch_size) else 0

        padded = cv2.copyMakeBorder(
            image,
            0,
            self.extra_h,
            0,
            self.extra_w,
            cv2.BORDER_CONSTANT,
            value=0
        )
        return padded

    def unpad_image(self, pred_full, original_height, original_width):
        """
        Remove the extra bottom/right padding so shape matches the cropped region size.
        """
        return pred_full[:original_height, :original_width]

    ###########################################################################
    # Conversions: (row, col) => mm => robot
    ###########################################################################
    def convert_to_mm(self, pixel_coords, conversion_factor):
        """
        row => mm_x
        col => mm_y
        We also add the crop offsets: row+ self.crop_top, col+ self.crop_left
        """
        result = []
        for (r, c) in pixel_coords:
            mm_x = (self.crop_top + r) * conversion_factor
            mm_y = (self.crop_left + c) * conversion_factor
            result.append((mm_x, mm_y))
        return result
    


    def convert_to_robot_coordinates(self, mm_coords):
        """
        mm_coords => (mm_x, mm_y)
        plate_position => self.plate_origin_in_robot => (rx, ry, rz)
        """
        result = []
        for (mm_x, mm_y) in mm_coords:
            x_robot = self.plate_origin_in_robot[0] + mm_x / 1000.0
            y_robot = self.plate_origin_in_robot[1] + mm_y / 1000.0
            z_robot = self.plate_origin_in_robot[2]  # e.g. keep pipeline's Z
            result.append((x_robot, y_robot, z_robot))
        return result

    ###########################################################################
    # Pipeline Steps
    ###########################################################################
    def process_image(self, image_path):
        """
        1) load grayscale
        2) detect & crop dish
        3) pad => patchify => run model => unpatchify => unpad
        Returns: final float mask (cropped shape), and the (cropped_h, cropped_w).
        """
        gray = self.load_grayscale_image(image_path)
        if gray is None:
            raise ValueError("[ERROR] Could not load or process image (None).")

        # detect dish edges => store left, top
        edges = self.detect_edges(gray)
        cropped = self.crop_image(gray, edges)

        cropped_h, cropped_w = cropped.shape

        padded = self.pad_image(cropped)
        p_h, p_w = padded.shape

        # patchify
        patches = patchify(padded, (self.patch_size, self.patch_size), step=self.patch_size)
        patches_reshaped = patches.reshape(-1, self.patch_size, self.patch_size, 1) / 255.0

        # inference
        preds = self.model.predict(patches_reshaped, batch_size=4)

        # unpatchify
        rows = p_h // self.patch_size
        cols = p_w // self.patch_size
        preds_reshaped = preds.reshape(rows, cols, self.patch_size, self.patch_size)
        pred_full = unpatchify(preds_reshaped, (p_h, p_w))

        # unpad => shape = cropped_h, cropped_w
        pred_unpadded = self.unpad_image(pred_full, cropped_h, cropped_w)

        return pred_unpadded, (cropped_h, cropped_w)

    def postprocess_and_extract(self, pred_float):
        """
        Threshold => remove small => morphological => skeleton => largest 5 => bottom endpoints.
        """
        bin_mask = (pred_float > 0.5).astype(np.uint8)
        bool_mask = bin_mask > 0
        cleaned_bool = remove_small_objects(bool_mask, min_size=self.morph_min_size)
        cleaned_mask = cleaned_bool.astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_kernel_size, self.morph_kernel_size))
        dilated = cv2.dilate(cleaned_mask, kernel, iterations=self.morph_dilate_iter)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=self.morph_close_iter)

        final_mask = (closed > 0).astype(np.uint8)*255
        skeleton_bool = skeletonize(closed > 0)

        skel_data = Skeleton(skeleton_bool)
        branch_data = summarize(skel_data)

        # find largest 5 skeletons
        skeleton_sizes = []
        for skel_id, grp in branch_data.groupby("skeleton-id"):
            total_length = grp["branch-distance"].sum()
            skeleton_sizes.append((skel_id, total_length))

        skeleton_sizes.sort(key=lambda x: x[1], reverse=True)
        top_5_ids = [sid for (sid, _) in skeleton_sizes[:5]]

        endpoints = []
        for sid in top_5_ids:
            grp = branch_data[branch_data["skeleton-id"] == sid]
            # bottom endpoint => largest row
            end_branch = grp.loc[
                grp[["image-coord-src-0", "image-coord-dst-0"]].max(axis=1).idxmax()
            ]
            max_y = end_branch[["image-coord-src-0", "image-coord-dst-0"]].max()
            if max_y == end_branch["image-coord-src-0"]:
                max_x = end_branch["image-coord-src-1"]
            else:
                max_x = end_branch["image-coord-dst-1"]

            # we store (row, col)
            endpoints.append((max_y, max_x))

        return final_mask, skeleton_bool, endpoints

    def run_pipeline(self, image_path, visualize=False):
        """
        Full pipeline:
        1) process_image => returns unpadded float mask + cropped shape
        2) morphological => endpoints in row/col
        3) convert (row, col) => mm => robot coords
        """
        pred_float, (crop_h, crop_w) = self.process_image(image_path)
        final_mask, skeleton_bool, endpoints_pixels = self.postprocess_and_extract(pred_float)

        

        # compute conversion factor
        conversion_factor = self.plate_size_mm / crop_w
        # convert (row, col) => mm => robot
        mm_coords = self.convert_to_mm(endpoints_pixels, conversion_factor)
        robot_coords = self.convert_to_robot_coordinates(mm_coords)

        return final_mask, skeleton_bool, endpoints_pixels, robot_coords

