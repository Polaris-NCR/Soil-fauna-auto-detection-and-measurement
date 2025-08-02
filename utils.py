import glob
import math
import os
import random
from pathlib import Path
import os
import joblib
import pandas as pd
import skimage
from matplotlib import pyplot as plt
from openpyxl import load_workbook
from torch import nn
from tqdm import tqdm
import cv2
import numpy as np
import torch
from itertools import combinations
from sklearn.metrics import average_precision_score
from osgeo import gdal_array as ga
from osgeo import gdal
from skimage.draw import disk, circle_perimeter
from collections import deque
import matplotlib.patches as patches
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from skimage.morphology import skeletonize, thin, medial_axis, binary_erosion, binary_opening, binary_dilation
from scipy.ndimage import label, generate_binary_structure
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splprep, splev
from sklearn.decomposition import PCA
import PIL
import modules, module1
from openpyxl.styles import Font
import torch.nn.functional as F


def Build_dataloader(image_array, mask_array, batch_num=32, weight1=0.6, weight2=0.1):
    image_array = image_array.astype(np.float32)
    mask_array = mask_array.astype(np.float32)
    mask_array = mask_array.reshape(mask_array.shape[0], 1, mask_array.shape[1], mask_array.shape[2])
    print(mask_array.shape)
    list1 = [i for i in range(1, image_array.shape[0])]
    random.shuffle(list1)
    train_num = int(image_array.shape[0] * weight1)
    val_num = int(image_array.shape[0] * weight2)
    test_num = image_array.shape[0] - train_num - val_num
    Xtrain = image_array[:train_num]
    Ytrain = mask_array[:train_num]
    Xval = image_array[train_num:train_num + val_num]
    Yval = mask_array[train_num:train_num + val_num]
    Xtest = image_array[train_num + val_num:]
    Ytest = mask_array[train_num + val_num:]
    print(Xtrain.shape, Ytrain.shape, Xval.shape, Yval.shape, Xtest.shape, Ytest.shape)
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(Xtrain), torch.from_numpy(Ytrain))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_num, shuffle=True)
    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(Xval), torch.from_numpy(Yval))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_num, shuffle=True)
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(Xtest), torch.from_numpy(Ytest))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_num, shuffle=True)
    return train_loader, val_loader, test_loader


def cal_scale(image_path):
    img = gdal.Open(image_path)
    raster = img.ReadAsArray()[0]
    scale_list = []
    for i in range(raster.shape[0]):
        list = raster[i]
        if np.any(list == 255):
            indices = np.argwhere(list == 255)
            start, end, index = 0, 0, 0
            for j in range(len(indices) - 1):
                if start == 0:
                    if indices[j] + 1 != indices[j + 1]:
                        start = indices[j]
                        index = j
            for k in range(index, len(indices) - 1):
                if end == 0:
                    if indices[k] + 1 != indices[k + 1]:
                        end = indices[j]
            scale = end - start + 1
            if scale > 1:
                scale_list.append(scale)
    scale_list = np.array(scale_list)
    scale = np.max(scale_list)
    print('scale', scale)
    return scale


def compute_direction_vector(arr):
    points = np.argwhere(arr == 255)
    if len(points) < 2:
        return None
    pca = PCA(n_components=1)
    pca.fit(points)
    print(pca.components_)
    direction_vector = pca.components_[0]
    return direction_vector


def compute_centroid(arr):
    total_mass = np.sum(arr)
    if total_mass == 0:
        return None
    centroid_y, centroid_x = np.argwhere(arr > 0).mean(axis=0)
    return centroid_x, centroid_y


def compute_center_of_mass(arr):
    return compute_centroid(arr)


def compute_symmetric_center(arr):
    arr_flipped_vertically = np.flipud(arr)
    arr_flipped_horizontally = np.fliplr(arr)

    vertical_diff = np.sum(np.abs(arr - arr_flipped_vertically))

    horizontal_diff = np.sum(np.abs(arr - arr_flipped_horizontally))

    if vertical_diff < horizontal_diff:

        symmetric_center = (arr.shape[1] // 2, arr.shape[0] // 2)
    else:

        symmetric_center = (arr.shape[1] // 2, arr.shape[0] // 2)

    return symmetric_center


def RGB_to_gray(raster, label_filter=False, savepath=None):
    if label_filter:
        array = raster[0]
        array[array == 255] = 0
        raster[0] = array

    raster_gray = 0.299 * raster[0] + 0.587 * raster[1] + 0.114 * raster[2]
    raster_gray = np.uint8(raster_gray)
    if savepath is not None:
        ga.SaveArray(raster_gray, savepath)
    return raster_gray


def build_circle_musk(gray, savepath=None):
    height, width = gray.shape[0], gray.shape[1]
    center = (height // 2, width // 2)
    index = np.argmax(gray[0] > 110)
    if index != 0:
        radius = np.sqrt(np.power(center[0], 2) + np.power((center[1] - index), 2))
    else:
        radius = 960

    radius = math.floor(radius)

    rr, cc = disk(center, radius, shape=(height, width))
    rr1, cc1 = disk(center, radius + 10, shape=(height, width))
    distances = np.sqrt((rr1 - center[0]) ** 2 + (cc1 - center[1]) ** 2)
    valid_indices = (distances >= radius - 30) & (distances <= radius + 30)

    mask = np.zeros((height, width), dtype=bool)
    mask[rr, cc] = True
    cropped_image = np.zeros_like(gray)
    cropped_image[mask] = gray[mask]
    if savepath is not None:
        ga.SaveArray(cropped_image, savepath)

    return cropped_image


def get_open_image(binary_image, size=5, savepath=None):
    kernel = np.ones((size, size), np.uint8)
    opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    opened_image[opened_image == 0] = 100
    opened_image[opened_image == 255] = 0
    opened_image[opened_image == 100] = 255
    if savepath is not None:
        ga.SaveArray(opened_image, savepath)
    return opened_image


def get_binary(musked_image, savepath=None):
    blur = cv2.GaussianBlur(musked_image, (7, 7), 0)
    binary_image = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 2)
    height, width = binary_image.shape[0], binary_image.shape[1]
    center = (height // 2, width // 2)
    index = np.argmax(binary_image[0] == 0)
    radius = np.sqrt(np.power(center[0], 2) + np.power((center[1] - index), 2))
    radius = math.floor(radius)
    x_min = center[1] - radius
    x_max = center[1] + radius

    if savepath is not None:
        ga.SaveArray(binary_image, savepath)
    return x_min, x_max, binary_image


def gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2

    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel /= 2 * np.pi * sigma ** 2
    kernel /= np.sum(kernel)
    return kernel


def Gaussian_Filtering(img, kernel_size, sigma):
    kernel = gaussian_kernel(kernel_size, sigma)
    if len(img.shape) == 3:
        height, width, channels = img.shape
    else:
        height, width = img.shape
        channels = 1
    result = np.zeros_like(img, dtype=np.float32)

    pad_size = kernel_size // 2
    img_pad = np.pad(img, [(pad_size, pad_size), (pad_size, pad_size), (0, 0)], mode='constant')
    for c in range(channels):
        for i in range(pad_size, height + pad_size):
            for j in range(pad_size, width + pad_size):
                result[i - pad_size, j - pad_size, c] = np.sum(
                    kernel * img_pad[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1, c])

    return np.uint8(result)


def crop_image_and_pre(img_path, module, device, crop_size=64, stride=64, output_dir=None):
    img = gdal.Open(img_path)
    raster = img.ReadAsArray()
    if raster.shape[0] == 4:
        raster = raster[:3]

    indices = np.where(raster[0] == 255)
    gray = RGB_to_gray(raster, label_filter=True, savepath=None)

    h, w = raster.shape[1:]

    mask_arr = np.zeros_like(raster[0])
    out_arr = np.zeros_like(raster[0])

    for y in range(0, h, stride):
        for x in range(0, w, stride):

            if (y + crop_size) > h or (x + crop_size) > w:
                continue

            img_crop = raster[:, y:y + crop_size, x:x + crop_size]

            tensor = torch.tensor(img_crop).float().unsqueeze(0).to(device)
            output = module(tensor)
            pred_mask = (output > 0.5).float().squeeze().cpu().numpy()
            pred_mask[pred_mask == 1.0] = 255
            mask_arr[y:y + crop_size, x:x + crop_size] = pred_mask

    mask_arr[indices[0], indices[1]] = 0
    height, width = mask_arr.shape[0], mask_arr.shape[1]
    center = (height // 2, width // 2)
    index = np.argmax(mask_arr[0] == 0)
    radius = np.sqrt(np.power(center[0], 2) + np.power((center[1] - index), 2))
    radius = math.floor(radius) - 100
    rr, cc = disk(center, radius, shape=(height, width))

    mask = np.zeros((height, width), dtype=bool)
    mask[rr, cc] = True
    out_arr[mask] = mask_arr[mask]
    raster = raster.transpose(1, 2, 0)
    return raster, mask_arr, out_arr


def find_connected_components(binary_image, direction_num=4):
    rows, cols = binary_image.shape

    labels = np.zeros_like(binary_image, dtype=np.int32)
    current_label = 1

    if direction_num == 4:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if direction_num == 8:
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1), (0, 1),
                      (1, -1), (1, 0), (1, 1)]

    for i in range(rows):
        for j in range(cols):

            if binary_image[i, j] != 0 and labels[i, j] == 0:

                queue = deque()
                queue.append((i, j))
                labels[i, j] = current_label

                while queue:
                    x, y = queue.popleft()
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy

                        if 0 <= nx < rows and 0 <= ny < cols:
                            if binary_image[nx, ny] != 0 and labels[nx, ny] == 0:
                                labels[nx, ny] = current_label
                                queue.append((nx, ny))

                current_label += 1

    return labels


def calculate_orientation_pca(y_coords, x_coords):
    points = np.column_stack((x_coords, y_coords))

    cov_matrix = np.cov(points, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    main_vector = eigenvectors[:, np.argmax(eigenvalues)]

    angle_rad = np.arctan2(main_vector[1], main_vector[0])
    angle_deg = np.degrees(angle_rad)

    return eigenvalues, angle_deg


def calculate_region_properties(labels):
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]

    properties = {}

    for label in unique_labels:
        y_coords, x_coords = np.where(labels == label)

        area = len(y_coords)

        min_y, max_y = np.min(y_coords), np.max(y_coords)
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        bbox_height = max_y - min_y + 1
        bbox_width = max_x - min_x + 1

        centroid_y = (min_y + max_y) / 2.0
        centroid_x = (min_x + max_x) / 2.0

        if len(y_coords) > 1:
            eigenvalues, orientation = calculate_orientation_pca(y_coords, x_coords)
        else:
            orientation = 0
            eigenvalues = []

        properties[label] = {
            'area': area,
            'bbox': (min_x, max_x, min_y, max_y),
            'rect_points': (min_x, min_y, max_x, max_y),
            'bbox_width': bbox_width,
            'bbox_height': bbox_height,
            'centroid': (centroid_y, centroid_x),
            'orientation': orientation,
            'eigenvalues': eigenvalues
        }

    return properties


def calculate_angle(vec1, vec2):
    dot_product = np.dot(vec1, vec2)

    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)

    cos_theta = dot_product / (magnitude_vec1 * magnitude_vec2)

    angle_radians = math.acos(cos_theta)

    angle_degrees = math.degrees(angle_radians)

    return angle_degrees


def find_rect_and_remove(arr):
    labeled = find_connected_components(arr, direction_num=8)
    properties = calculate_region_properties(labeled)
    area_list = []
    rect_list = []
    label_list = []

    for label, props in properties.items():
        label_list.append(label)
        area_list.append(props['area'])
        rect_list.append(props['bbox'])
    if len(area_list) == 1:
        return arr
    else:
        max_index = np.argmax(area_list)
        label_dex = label_list[max_index]
        filter_arr = np.zeros_like(arr)
        filter_arr[labeled == label_dex] = 255
        return filter_arr


def find_hole_and_fill(arr, size=5):
    kernel_size = (size, size)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closing = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)
    return closing


def find_endpoints(skel):
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)
    conv = cv2.filter2D(skel, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    return (skel == 255) & ((conv == 11) | (conv == 12))


def prune_branches(skel, max_length=5):
    pruned = skel.copy()
    endpoints = find_endpoints(pruned)
    rows, cols = np.where(endpoints)

    for r, c in zip(rows, cols):
        branch = []
        current = (r, c)
        prev = (-1, -1)
        for _ in range(max_length):
            neighbors = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = current[0] + dr, current[1] + dc
                    if 0 <= nr < skel.shape[0] and 0 <= nc < skel.shape[1]:
                        if pruned[nr, nc] == 255 and (nr, nc) != prev:
                            neighbors.append((nr, nc))
            if len(neighbors) != 1:
                break
            prev = current
            current = neighbors[0]
            branch.append(current)
        if len(branch) <= max_length:
            for (br, bc) in branch:
                pruned[br, bc] = 0
    return pruned


def prune_skeleton(skeleton, min_branch_length=10):
    pruned = skeleton.copy()
    endpoints = []
    height, width = pruned.shape

    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1), (0, 1),
                 (1, -1), (1, 0), (1, 1)]

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if pruned[y, x] == 255:
                count = 0
                for dy, dx in neighbors:
                    if pruned[y + dy, x + dx] == 255:
                        count += 1
                if count == 1:
                    endpoints.append((y, x))
    print('endpoints', endpoints)

    for y, x in endpoints:
        branch = []
        current = (y, x)
        prev = (-1, -1)

        while True:
            branch.append(current)
            next_points = []
            for dy, dx in neighbors:
                ny = current[0] + dy
                nx = current[1] + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if pruned[ny, nx] == 255 and (ny, nx) != prev:
                        next_points.append((ny, nx))

            if len(next_points) != 1:
                break

            prev = current
            current = next_points[0]

        if len(branch) <= min_branch_length:
            for py, px in branch:
                pruned[py, px] = 0

    return pruned


def fit_PCA_of_arr(img, visualize=False):
    height, width = img.shape

    y_coords_img, x_coords_img = np.where(img == 255)
    if len(x_coords_img) < 2:
        raise ValueError("Error, require at least two point")

    y_coords_cart = height - y_coords_img - 1
    points = np.column_stack((x_coords_img, y_coords_cart))

    mean = np.mean(points, axis=0)
    centered = points - mean

    cov_matrix = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    max_idx = np.argmax(eigenvalues)
    vx, vy = eigenvectors[:, max_idx]

    std_dev = np.sqrt(eigenvalues[max_idx])
    scale = 3 * std_dev

    point_start = mean + scale * np.array([vx, vy])
    point_end = mean - scale * np.array([vx, vy])

    def cart2img(point):
        return int(round(point[0])), int(round(height - point[1] - 1))

    pt1 = cart2img(point_start)
    pt2 = cart2img(point_end)

    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.line(color_img, pt1, pt2, (0, 0, 255), 2)
    if visualize:
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
        plt.title('Principal Component Direction')
        plt.axis('off')
        plt.show()

    if abs(vx) < 1e-6:
        equation = f"x = {mean[0]:.2f}"
        return 0, 90
    else:
        slope = vy / vx
        intercept = mean[1] - slope * mean[0]
        equation = f"y = {slope:.2f}x + {intercept:.2f}"
        print('slope', slope, 'intercept', intercept, "PCA equation:", equation)
        angle = math.degrees(math.atan(slope))
        if angle < 0:
            angle += 180
        return slope, intercept, angle


def fit_curve_of_arr(img, degree=1):
    height, width = img.shape

    y_img, x_coords = np.where(img == 255)

    if len(x_coords) < 3:
        raise ValueError("Unable to perform quadratic polynomial fitting")

    y_cart = height - y_img - 1

    coefficients = np.polyfit(x_coords, y_cart, degree)
    poly = np.poly1d(coefficients)

    x_fit = np.linspace(min(x_coords), max(x_coords), 100)
    y_fit_cart = poly(x_fit)
    y_fit_img = height - y_fit_cart - 1

    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(len(x_fit) - 1):
        pt1 = (int(round(x_fit[i])), int(round(y_fit_img[i])))
        pt2 = (int(round(x_fit[i + 1])), int(round(y_fit_img[i + 1])))
        cv2.line(color_img, pt1, pt2, (0, 0, 255), 2)

    terms = []
    for i, coef in enumerate(coefficients):
        power = degree - i
        if power == 0:
            terms.append(f"{coef:.2f}")
        else:
            terms.append(f"{coef:.2f}x^{power}" if power > 1 else f"{coef:.2f}x")
    equation = "y = " + " + ".join(terms).replace("+ -", "- ")

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
    plt.title('Curve Fitting Result')
    plt.axis('off')
    plt.show()

    print("Equation", equation)

def open_calculation(raster, size):
    struct_element = skimage.morphology.disk(size)
    eroded = binary_erosion(raster, struct_element)
    opened_image = binary_dilation(eroded, struct_element)
    opened_image = opened_image.astype(np.uint8) * 255
    return opened_image


def cal_arr_num(arr):
    labeled = find_connected_components(arr, direction_num=8)
    properties = calculate_region_properties(labeled)
    num = 0
    area_list = [v['area'] for v in properties.values()]
    area_list = np.array(area_list)
    for label, props in properties.items():
        if props['area'] >= np.mean(area_list) / 4:
            num += 1
    return num


def find_rect_new(image_path, model, device, size=11, max_size=25, area_threhold=500):
    model = model.to(device)
    image_raster, mask_arr, raster = crop_image_and_pre(image_path, model, device)
    labeled = find_connected_components(raster, direction_num=8)
    properties = calculate_region_properties(labeled)
    raster_copy = raster.copy()
    area_list1 = [v['area'] for v in properties.values()]
    area_list1 = np.array(area_list1)
    for label, props in properties.items():
        if props['area'] >= np.mean(area_list1) / 4:
            h = min(props['bbox'][3] - props['bbox'][2] + 1, props['bbox'][1] - props['bbox'][0] + 1)
            arr = raster[props['bbox'][2]:props['bbox'][3] + 1, props['bbox'][0]:props['bbox'][1] + 1].copy()

            num1 = cal_arr_num(arr)
            if h // 10 < 16:
                window_size = size
            else:
                window_size = min(h // 10, max_size)

            arr1 = open_calculation(arr.copy(), window_size)
            num2 = cal_arr_num(arr1)
            if num2 == 0:
                arr1 = open_calculation(arr.copy(), size)
            if num2 > num1:
                arr2 = open_calculation(arr.copy(), window_size - 5)
                num3 = cal_arr_num(arr2)
                if num1 == num3:
                    arr1 = arr2
                elif num3 > num1:
                    arr1 = open_calculation(arr.copy(), window_size)
                else:
                    arr1 = open_calculation(arr.copy(), size)
            raster_copy[props['bbox'][2]:props['bbox'][3] + 1, props['bbox'][0]:props['bbox'][1] + 1] = arr1

        else:
            raster_copy[props['bbox'][2]:props['bbox'][3] + 1, props['bbox'][0]:props['bbox'][1] + 1].fill(0)

    labeled2 = find_connected_components(raster_copy, direction_num=8)
    properties2 = calculate_region_properties(labeled2)

    return image_raster, raster, raster_copy, properties, properties2


def is_overlapping(rect1, rect2):
    return not (rect1[2] <= rect2[0] or rect1[0] >= rect2[2] or rect1[3] <= rect2[1] or rect1[1] >= rect2[3])



def is_contained(rect1, rect2):
    return rect1[0] <= rect2[0] and rect1[1] <= rect2[1] and rect1[2] >= rect2[2] and rect1[3] >= rect2[3]


def is_crossing(rect1, rect2):
    if is_overlapping(rect1, rect2):
        if not is_contained(rect1, rect2) and not is_contained(rect2, rect1):
            return True
    return False


def get_intersection(rect1, rect2):
    xmin = max(rect1[0], rect2[0])
    ymin = max(rect1[1], rect2[1])
    xmax = min(rect1[2], rect2[2])
    ymax = min(rect1[3], rect2[3])
    if xmin < xmax and ymin < ymax:
        return (xmin, ymin, xmax, ymax)
    else:
        return None


def check_matrix_relations(matrices):
    relations = {}
    n = len(matrices)

    for i in range(n):
        for j in range(i + 1, n):
            rect1 = matrices[i]
            rect2 = matrices[j]
            relation = None

            if is_contained(rect1, rect2):
                relation = f"{i + 1},{j + 1},contain"
                relations[(i + 1, j + 1)] = (relation, rect2)
            elif is_contained(rect2, rect1):
                relation = f"{j + 1},{i + 1},contain"
                relations[(j + 1, i + 1)] = (relation, rect1)
            elif is_crossing(rect1, rect2):
                relation = f"{i + 1},{j + 1},cross"
                intersection = get_intersection(rect1, rect2)
                relations[(i + 1, j + 1)] = (relation, intersection)
            elif is_overlapping(rect1, rect2):
                relation = f"{i + 1},{j + 1},overlap"
                relations[(i + 1, j + 1)] = (relation, rect1)

    return relations

def build_adjacency(image):
    adj = {}
    height, width = image.shape
    for y in range(height):
        for x in range(width):
            if image[y, x] == 255:
                neighbors = []
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    ny = y + dy
                    nx = x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        if image[ny, nx] == 255:
                            neighbors.append((nx, ny))
                adj[(x, y)] = neighbors
    return adj


def remove_endpoints_from_adj(adj, removed_endpoints):
    new_adj = {k: v.copy() for k, v in adj.items()}

    for ep in removed_endpoints:
        if ep in new_adj:
            for neighbor in new_adj[ep]:
                if neighbor in new_adj:
                    new_adj[neighbor] = [n for n in new_adj[neighbor] if n != ep]

            del new_adj[ep]

    return new_adj


def find_endpoints(adj):
    return [point for point, neighbors in adj.items() if len(neighbors) == 1]


def bfs_furthest(start, adj):
    visited = {start: None}
    queue = deque([start])
    furthest_node = start

    while queue:
        current = queue.popleft()
        for neighbor in adj.get(current, []):
            if neighbor not in visited:
                visited[neighbor] = current
                queue.append(neighbor)
                furthest_node = neighbor  # 更新最远节点
    return furthest_node, visited


def get_path_between_two_endpoints(adj, start, end):
    if start not in adj or end not in adj:
        return None

    visited = {}
    queue = deque([start])
    visited[start] = None

    while queue:
        current = queue.popleft()

        if current == end:
            path = []
            while current is not None:
                path.append(current)
                current = visited[current]
            return path[::-1]

        for neighbor in adj.get(current, []):
            if neighbor not in visited:
                visited[neighbor] = current
                queue.append(neighbor)

    return None


def get_longest_path_from_rect_endpoints(adj, endpoints):
    max_length = 0
    best_path = None
    best_start = None
    best_end = None
    for (x1, y1), (x2, y2) in combinations(endpoints, 2):
        path = get_path_between_two_endpoints(adj, (x1, y1), (x2, y2))
        if len(path) > max_length:
            best_path = path
            best_start = (x1, y1)
            best_end = (x2, y2)
    return best_start, best_end, best_path


def get_longest_path(adj, endpoints):
    if len(endpoints) < 2:
        return None, None, []

    start = endpoints[0]
    u, _ = bfs_furthest(start, adj)

    v, visited = bfs_furthest(u, adj)

    path = []
    current = v
    while current is not None:
        path.append(current)
        current = visited.get(current)
    path.reverse()

    return u, v, path


def calculate_slope(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2
    dx = x2 - x1
    dy = y2 - y1

    if dx == 0:
        slope = np.inf
        angle = np.pi / 2
    else:
        slope = dy / dx
        angle = np.arctan2(dy, dx)

    return {
        "slope_value": slope,
        "angle_rad": angle,
        "angle_deg": np.degrees(angle) % 360
    }


def get_all_paths(adj):
    endpoints = [point for point, neighbors in adj.items() if len(neighbors) == 1]

    paths = []
    visited = set()

    for start in endpoints:
        queue = deque([(start, [start])])

        while queue:
            current, path = queue.popleft()

            if current in endpoints and current != start and tuple(sorted([start, current])) not in visited:
                slope = calculate_slope(path[0], path[-1])
                paths.append({
                    "points": path.copy(),
                    "slope": slope,
                    "length": len(path)
                })
                visited.add(tuple(sorted([start, current])))  # 避免重复记录A-B和B-A

            for neighbor in adj.get(current, []):
                if neighbor not in path:
                    queue.append((neighbor, path + [neighbor]))

    return paths


def trans_path_matrix(original_image, path_points):
    path_matrix = np.zeros_like(original_image)
    for (x, y) in path_points:
        path_matrix[y, x] = 255
    return path_matrix


def calculate_skeleton_length(path):
    if len(path) < 2:
        return 0.0, 0

    geo_length = 0.0
    for i in range(1, len(path)):
        x1, y1 = path[i - 1]
        x2, y2 = path[i]
        geo_length += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    pixel_count = len(path)

    return geo_length, pixel_count


def visualize_result(image, path, point1, point2):
    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for i in range(1, len(path)):
        cv2.line(result, path[i - 1], path[i], (0, 0, 255), 2)
    cv2.circle(result, point1, 5, (0, 255, 0), -1)
    cv2.circle(result, point2, 5, (0, 255, 0), -1)
    return result

def visualize_axis(original_image, path_points, pos_point, neg_point, value=110):
    image = np.zeros_like(original_image)
    original_image = draw_width_bresenham(image, pos_point, neg_point, value=110)
    # original_image = draw_width_bresenham(original_image, pos_point, neg_point, value=110)
    for (x, y) in path_points:
        original_image[y, x] = value  # 注意坐标转换

    return original_image


def draw_width_bresenham(img, pt1, pt2, value=110):
    x0, y0 = pt1
    x1, y1 = pt2
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        if 0 <= x0 < img.shape[1] and 0 <= y0 < img.shape[0]:
            img[y0, x0] = value
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return img

def extend_endpoint(img_binary, endpoint, direction_vector, max_steps=1000):
    height, width = img_binary.shape
    x, y = endpoint
    dx, dy = direction_vector
    path = [endpoint]

    length = np.hypot(dx, dy)
    if length > 0:
        dx /= length
        dy /= length

    for _ in range(max_steps):
        x += dx
        y += dy
        xi, yi = int(round(x)), int(round(y))

        if xi < 0 or xi >= width or yi < 0 or yi >= height:
            return True, path

        if img_binary[yi, xi] != 255:
            return True, path

        path.append((int(x), int(y)))

    return False, path


def get_tangent_direction(skeleton, point, lookback=15):
    x, y = point
    points = [(x, y)]

    for _ in range(lookback):
        found = False
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx = x + dx
                ny = y + dy
                if (nx, ny) in points:
                    continue
                if 0 <= nx < skeleton.shape[1] and 0 <= ny < skeleton.shape[0]:
                    if skeleton[ny, nx] == 255:
                        x, y = nx, ny
                        points.append((x, y))
                        found = True
                        break
            if found:
                break
        if not found:
            break

    if len(points) >= 2:
        x0, y0 = points[0]
        x1, y1 = points[-1]
        dx = x1 - x0
        dy = y1 - y0
        length = np.hypot(dx, dy)
        if length > 0:
            return dx / length, dy / length
    return 1, 0


def extend_skeleton_ends(orig_binary, skeleton, original_path, lookback=5):
    if len(original_path) < 1:
        return original_path, []

    start_point = original_path[0]
    end_point = original_path[-1]

    start_dir = get_tangent_direction(skeleton, start_point, lookback)
    end_dir = get_tangent_direction(skeleton, end_point, lookback)

    reverse_start_dir = (-start_dir[0], -start_dir[1])
    reverse_end_dir = (-end_dir[0], -end_dir[1])

    start_success, start_path = extend_endpoint(orig_binary, start_point, reverse_start_dir)
    end_success, end_path = extend_endpoint(orig_binary, end_point, reverse_end_dir)

    extended_path = []
    if start_success:
        extended_path += list(reversed(start_path))
    extended_path += original_path
    if end_success:
        extended_path += end_path[1:]

    new_ends = []
    if start_success and len(start_path) > 1:
        new_ends.append(start_path[-1])
    if end_success and len(end_path) > 1:
        new_ends.append(end_path[-1])

    return extended_path, new_ends


def smooth_skeleton(points, window_size=5):
    smoothed = []
    n = len(points)
    for i in range(n):
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2 + 1)
        window = points[start:end]
        avg_x = int(sum(p[0] for p in window) / len(window))
        avg_y = int(sum(p[1] for p in window) / len(window))
        smoothed.append((avg_x, avg_y))
    return smoothed


def smooth_skeleton_PCA(points, PCA_slope, window_size=5):
    smoothed = []
    points = [list(p) for p in points]
    n = len(points)
    for i in range(n):
        start = max(0, i - window_size // 2)
        end = min(n - 1, i + window_size // 2 + 1)
        # print('start', start, 'end', end)
        if points[start][0] - points[end][0] != 0:
            k = -(points[start][1] - points[end][1]) / (points[start][0] - points[end][0])
            error = abs(k - PCA_slope) / PCA_slope

            if error > 0.1 and PCA_slope > 0:
                if k < PCA_slope:
                    points[start][0] += 1
                    # points[start][1] += 1
                elif k > PCA_slope:
                    points[start][0] -= 1
                    # points[start][1] -= 1
            if error > 0.1 and PCA_slope < 0:
                if k < PCA_slope:
                    # points[start][0] += 1
                    points[start][1] -= 1
                elif k > PCA_slope:
                    # points[start][0] -= 1
                    points[start][1] += 1
        else:
            if PCA_slope > 0:
                points[start][0] -= 1
            elif PCA_slope < 0:
                points[start][0] += 1
        window = points[start:end]
        avg_x = int(sum(p[0] for p in window) / len(window))
        avg_y = int(sum(p[1] for p in window) / len(window))
        smoothed.append([avg_x, avg_y])

    smoothed = tuple(tuple(sub_list) for sub_list in smoothed)

    return smoothed


def constrained_smooth(binary_image, raw_path, kernal_size=5, max_iter=10):
    smoothed = np.array(raw_path)
    height, width = binary_image.shape

    for _ in range(max_iter):
        smoothed = cv2.GaussianBlur(smoothed.astype(float), (kernal_size, kernal_size), 0)
        for i in range(len(smoothed)):
            x, y = smoothed[i]
            min_dist = float('inf')
            best = (x, y)
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    px = int(x) + dx
                    py = int(y) + dy
                    if 0 <= px < width and 0 <= py < height:
                        if binary_image[py, px] == 255:
                            dist = (dx ** 2 + dy ** 2)
                            if dist < min_dist:
                                min_dist = dist
                                best = (px, py)
            smoothed[i] = best
    return smoothed.tolist()


def smooth_skeleton_to_array(points, arr):
    smoothed_binary = float_to_binary(points, arr.shape)
    skeleton = skeletonize(smoothed_binary // 255)
    return skeleton.astype(np.uint8) * 255


def float_to_binary(points, image_shape, line_thickness=1):
    binary = np.zeros(image_shape, dtype=np.uint8)
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

    if line_thickness > 1:
        binary = cv2.polylines(binary, [pts], False, 255, thickness=line_thickness, lineType=cv2.LINE_AA)
    else:
        for i in range(len(points) - 1):
            pt1 = tuple(np.round(points[i]).astype(int))
            pt2 = tuple(np.round(points[i + 1]).astype(int))

            line_points = bresenham_line(pt1, pt2)
            for (x, y) in line_points:
                if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                    binary[y, x] = 255

    return binary


def bresenham_line(start, end):
    x1, y1 = start
    x2, y2 = end
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    steep = dy > dx

    if steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
        dx, dy = dy, dx

    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1

    dx = x2 - x1
    dy = abs(y2 - y1)
    error = dx // 2
    ystep = 1 if y1 < y2 else -1
    y = y1
    line = []

    for x in range(x1, x2 + 1):
        coord = (y, x) if steep else (x, y)
        line.append(coord)
        error -= dy
        if error < 0:
            y += ystep
            error += dx
    return line

