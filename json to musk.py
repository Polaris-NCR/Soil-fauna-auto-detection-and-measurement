import os
import json
import numpy as np
from PIL import Image
import cv2


def json_to_mask(json_path, output_dir, class_mapping):
    with open(json_path, 'r') as f:
        data = json.load(f)

    img_width = data['imageWidth']
    img_height = data['imageHeight']
    print(img_width, img_height)
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    for shape in data['shapes']:
        label = shape['label']
        print('label', label)
        points = shape['points']
        shape_type = shape.get('shape_type', 'polygon')

        if label not in class_mapping:
            continue 
        class_idx = class_mapping[label]

        pts = np.array(points, dtype=np.int32)

        if shape_type == 'polygon':
            cv2.fillPoly(mask, [pts], color=class_idx)

    mask_img = Image.fromarray(mask)
    output_path = os.path.join(output_dir, os.path.basename(json_path).replace('.json', '.png'))
    mask_img.save(output_path)
    print(f'Saved mask to {output_path}')


if __name__ == '__main__':
    json_dir = r'E:\T10'
    output_dir = r'E:\T10-json-mask' 
    class_mapping = {'nematode': 255}

    os.makedirs(output_dir, exist_ok=True)

    for json_file in os.listdir(json_dir):
        if json_file.endswith('.json'):
            json_path = os.path.join(json_dir, json_file)
            json_to_mask(json_path, output_dir, class_mapping)


