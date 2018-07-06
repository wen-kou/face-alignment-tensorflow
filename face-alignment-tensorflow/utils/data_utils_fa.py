import cv2
import os
import numpy as np
import pandas as pd

from utils import utils


def generate_sample_face(image, landmarks, detector, input_resolution=256, output_resolution=64):
    num_landmarks = len(landmarks)
    detected_faces = utils.get_face_bbox(image, detector)
    outputs = list()
    if len(detected_faces) > 0:
        for i, rect in enumerate(detected_faces):
            center = [(rect.left() + rect.right()) / 2,
                      (rect.top() + rect.bottom()) / 2]
            center[1] = center[1] - (rect.bottom() - rect.top()) * 0.12

            # scale = (rect.right() - rect.left() +
            #          rect.bottom() - rect.top()) / 195.0
            scale = 2.0

            cropped_image = utils.crop(image, center, scale, resolution=input_resolution)
            heatmaps = np.zeros((output_resolution, output_resolution, num_landmarks))

            transformed_landmarks = []
            for j in range(num_landmarks):
                ldmk = utils.transform(landmarks[j] + 1, center, scale, resolution=output_resolution)
                transformed_landmarks.append(ldmk)
                tmp = utils.draw_gaussian(heatmaps[:, :, j],
                                          ldmk,
                                          1)
                heatmaps[:, :, j] = tmp
            outputs.append({'image': cropped_image / 255, 'heatmaps': heatmaps, 'center': center,
                            'scale':scale, 'pts': transformed_landmarks})
    return outputs


def post_process(heatmaps, center, scale):
    '''
    :param heatmaps: ndarray shape{1, resolution, resolution, num_facial_landmarks}
    :param center: cropped center
    :param scale: scale factor
    :return: landmarks of cropped image and original image
    '''
    resolution_y, resolution_x = heatmaps.shape[1], heatmaps.shape[2]
    num_landmarks = heatmaps.shape[3]
    landmarks = []
    for i in range(num_landmarks):
        heatmap = heatmaps[0, :, :, i]
        indx = np.argmax(heatmap)
        coord = np.unravel_index(indx, (resolution_y, resolution_x))
        coord = [coord[1], coord[0]]
        landmarks.append(list(coord))

    landmarks_origins = []
    for i in range(num_landmarks):
        tmp = utils.transform(np.asarray(landmarks[i]) + 1, center, scale, resolution_x, True)
        landmarks_origins.append(tmp)

    return landmarks, landmarks_origins


# def get_preds_fromhm(hm, center=None, scale=None):
#     max, idx = torch.max(
#         hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
#     idx += 1
#     preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
#     preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
#     preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)
#
#     for i in range(preds.size(0)):
#         for j in range(preds.size(1)):
#             hm_ = hm[i, j, :]
#             pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
#             if pX > 0 and pX < 63 and pY > 0 and pY < 63:
#                 diff = torch.FloatTensor(
#                     [hm_[pY, pX + 1] - hm_[pY, pX - 1],
#                      hm_[pY + 1, pX] - hm_[pY - 1, pX]])
#                 preds[i, j].add_(diff.sign_().mul_(.25))
#
#     preds.add_(-.5)
#
#     preds_orig = torch.zeros(preds.size())
#     if center is not None and scale is not None:
#         for i in range(hm.size(0)):
#             for j in range(hm.size(1)):
#                 tmp = utils.transform(
#                     preds[i, j].numpy(), center, scale, hm.size(2), True)
#                 preds_orig[i, j] = torch.tensor(tmp)
#
#     return preds, preds_orig

