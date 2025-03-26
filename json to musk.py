import os
import json
import numpy as np
from PIL import Image
import cv2


def json_to_mask(json_path, output_dir, class_mapping):
    # 加载JSON文件
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 获取图像尺寸
    img_width = data['imageWidth']
    img_height = data['imageHeight']
    print(img_width, img_height)
    # 创建全零掩码（默认背景为0）
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # 遍历所有标注形状
    for shape in data['shapes']:
        label = shape['label']
        print('label', label)
        points = shape['points']
        shape_type = shape.get('shape_type', 'polygon')

        # 获取类别索引
        if label not in class_mapping:
            continue  # 忽略未定义的类别
        class_idx = class_mapping[label]

        # 将多边形点转换为OpenCV格式
        pts = np.array(points, dtype=np.int32)

        # 绘制形状到掩码（多边形用fillPoly，其他类型可扩展）
        if shape_type == 'polygon':
            cv2.fillPoly(mask, [pts], color=class_idx)
        # 可添加其他形状处理（如矩形、圆形）

    # 保存掩码图像
    mask_img = Image.fromarray(mask)
    output_path = os.path.join(output_dir, os.path.basename(json_path).replace('.json', '.png'))
    mask_img.save(output_path)
    print(f'Saved mask to {output_path}')


# 示例调用
if __name__ == '__main__':
    json_dir = r'E:\萌姐线虫数据\T10'  # JSON文件目录
    output_dir = r'E:\萌姐线虫数据\T10-json-mask'  # 输出掩码目录
    class_mapping = {'nematode': 255}  # 类别到索引的映射

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历所有JSON文件并生成掩码
    for json_file in os.listdir(json_dir):
        if json_file.endswith('.json'):
            json_path = os.path.join(json_dir, json_file)
            json_to_mask(json_path, output_dir, class_mapping)
