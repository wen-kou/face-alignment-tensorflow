import cv2
import os
import numpy as np
import pandas as pd

from utils import data_utils_fa
from multiprocessing.pool import ThreadPool


def get_training_batch(root, training_file_name, detector, batch_size=32, num_landmarks=76, input_resolution=256,
                       output_resolution=64):
    training_element = pd.read_csv(os.path.join(root, training_file_name)).sample(frac=1)
    # training_element = training_element.sample(frac=1, replace=True)
    training_num = len(training_element)
    batch_count = 0
    sample_inputs = np.zeros((batch_size, input_resolution, input_resolution, 3))
    sample_outputs = np.zeros((batch_size, output_resolution, output_resolution, num_landmarks))

    for global_count in range(training_num):
        item = training_element.iloc[global_count]

        image_path = item[0]
        image = cv2.imread(os.path.join(root, image_path))

        if image is None:
            # print('No image for {}th item with image name {} '.format(global_count, image_path))
            if global_count == training_num - 1:
                yield sample_inputs, sample_outputs
                return
            continue

        # print('Getting {}th item with image name {} '.format(global_count, image_path))
        landmarks = item[1:]
        tmp = []
        for i in range(0, len(landmarks), 2):
            landmark = [landmarks[i], landmarks[i + 1]]
            tmp.append(landmark)
        landmarks = np.asarray(tmp)
        samples = data_utils_fa.generate_sample_face(image, landmarks, detector)
        for sample in samples:
            if batch_count == batch_size:
                batch_count = 0
                inputs = sample_inputs
                outputs = sample_outputs
                sample_inputs = np.zeros((batch_size, input_resolution, input_resolution, 3))
                sample_outputs = np.zeros((batch_size, output_resolution, output_resolution, num_landmarks))
                yield inputs, outputs
            sample_input, sample_output = sample['image'], sample['heatmaps']
            sample_inputs[batch_count] = sample_input
            sample_outputs[batch_count] = sample_output
            batch_count += 1

        if global_count == training_num - 1:
            yield sample_inputs, sample_outputs
            return


def _get_landmarks(landmark_list):
    tmp = []
    for i in range(0, len(landmark_list), 2):
        landmark = [landmark_list[i], landmark_list[i + 1]]
        tmp.append(landmark)
    landmarks = np.asarray(tmp)
    return landmarks

global pool
pool = ThreadPool(128)


def _get_batch(root, training_items, detector):
    return pool.map(lambda training_item: _get_item(root, training_item, detector), training_items)


def _get_item(root, training_item, detector):
    image_path = os.path.join(root, training_item[0])
    image = cv2.imread(image_path)
    landmarks = _get_landmarks(training_item[1:])
    training_sample = data_utils_fa.generate_sample_face(image, landmarks, detector)
    return training_sample['cropped_image'],training_sample['heatmaps']