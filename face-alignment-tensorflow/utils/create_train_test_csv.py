import os
import pandas as pd

annotations_path = '../../muct/muct-landmarks/muct76-opencv.csv'
annotations = pd.read_csv(annotations_path).drop_duplicates()

image_root = '../../muct/dataset'

val_set = 'jpg_5'
training_images = []
image_name_list = []
for root, dir, filenames in os.walk(image_root):
    if (root == os.path.join(image_root, val_set)) & (root != image_root):
        for filename in filenames:
            training_images.append(os.path.join(root.split('/')[-1], filename))
            image_name_list.append(filename.split('.')[0])

training_items = []
annotations_columns = list(annotations)
annotations_columns.remove('tag')
for inx, item in annotations.iterrows():
    training_item = []
    name = item[0]
    if name in image_name_list:
        tmp = image_name_list.index(name)
        image_path = training_images[tmp]
        training_item.append(image_path)
        landmarks = item[2:]
        training_item.extend(landmarks)

        training_items.append(training_item)

training = pd.DataFrame(training_items, columns=annotations_columns).reset_index(drop=True)

training.to_csv('test.csv', index=False)