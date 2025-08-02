import math
import numpy as np
import torch
from osgeo import gdal
from skimage.draw import disk
import utils


def crop_image_and_pre(img_path, module, device, crop_size=64, stride=64, output_dir=None):
    img = gdal.Open(img_path)
    raster = img.ReadAsArray()
    if raster.shape[0] == 4:
        raster = raster[:3]

    indices = np.where(raster[0] == 255)
    gray = utils.RGB_to_gray(raster, label_filter=True, savepath=None)
    # musked_image = build_circle_musk(gray, savepath=None)

    h, w = raster.shape[1:]
    # base_name = os.path.splitext(os.path.basename(img_path))[0]

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


image_path=r'E:\\'
model = joblib.load(r'E:\\model.pkl')
image_raster, mask_arr, raster = crop_image_and_pre(image_path, model, device)