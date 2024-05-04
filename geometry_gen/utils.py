# vuer import
import os
from asyncio import sleep
from typing import List

# other imports
import numpy as np
import cv2
import PIL.Image as PImage
from io import BytesIO
import shutil


CONVERSION_INDICES = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]


def colmap2three(mat):
    """Converts a 4x4 colmap matrix to a three.js matrix"""
    return np.array(mat)[CONVERSION_INDICES]


def pos_to_rot(p, degrees=True):
    """Compute yaw and pitch of a position that points it towards the origin"""
    yaw = np.arctan2(p[1], p[0])
    pitch = np.arctan2(p[2], (p[0] * p[0] + p[1] * p[1]) ** (0.5))
    if degrees:
        return {'pitch': np.rad2deg(pitch), 'yaw': np.rad2deg(yaw)}
    else:
        return {'pitch': pitch, 'yaw': yaw}


def scale_extrinsics(extrinsics, scale):
    scaled_matrix = np.copy(extrinsics)
    scaled_matrix[:3, 3] *= scale
    return scaled_matrix


def save_render(buff, prefix, filename):
    # we send jpg, you can dump the file buffer directly to disk (as jpg).
    pil_image = PImage.open(BytesIO(buff))
    img = np.array(pil_image)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(prefix, filename), img_bgr)

def calculate_scale_factor(original_size, bounding_box):
    original_max_dimension = max(original_size)
    bounding_box_max_dimension = max(bounding_box)
    scale_factor = bounding_box_max_dimension / original_max_dimension
    return scale_factor

def empty_or_create_directory(path):
    if os.path.exists(path) and os.path.isdir(path):
        # Remove all contents of the directory
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        os.makedirs(path, exist_ok=True)

def create_dir_with_prompt(directory):
    if os.path.exists(directory):
        input(f"The directory {directory} already exists. Press Enter to overwrite.")
    
    empty_or_create_directory(directory)

