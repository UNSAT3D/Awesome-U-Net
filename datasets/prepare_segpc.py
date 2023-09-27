import gc
import glob
import json
import os
import pickle
import random
import sys
from configparser import ConfigParser
from pathlib import Path

import click
import cv2
import imageio.v2 as imageio
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from scipy import ndimage
from scipy.ndimage.morphology import (binary_dilation, binary_erosion,
                                      binary_fill_holes)
from tqdm import tqdm


HEIGHT = 1080
WIDTH = 1440
CHANNELS = 3
FILE_TYPE_EXTENSION = ".bmp"


# NOTE: this function wasn't defined, I made this up (using GPT4)
def resize_pad(img, target_size, pad_value=0):
    """
    Resizes the given image to the target size, padding it with pad_value if necessary.
    
    Args:
    img (np.ndarray): Input image.
    target_size (tuple): Desired (height, width).
    pad_value (int): Value to use for padding.
    
    Returns:
    np.ndarray: The resized and padded image.
    """
    height, width = img.shape[:2]
    target_height, target_width = target_size
    
    # Calculating scale and padding
    scale = min(target_height / height, target_width / width)
    new_height, new_width = int(height * scale), int(width * scale)
    pad_height, pad_width = (target_height - new_height) // 2, (target_width - new_width) // 2
    
    # Resizing the image
    resized_img = cv2.resize(img, (new_width, new_height))
    
    # Creating a new image of target size and pasting the resized image into it
    final_img = np.ones((target_height, target_width) + img.shape[2:], dtype=np.uint8) * pad_value
    final_img[pad_height:pad_height + new_height, pad_width:pad_width + new_width] = resized_img
    
    return final_img

# NOTE: this function wasn't defined, I made this up (using GPT4)
def sim_resize(img, target_size):
    return cv2.resize(img, (target_size[1], target_size[0]))


def split_c_n(img, nv=40, cv=20):
    nim = np.where(img >= nv, 1, 0)
    cim = np.where(img >= cv, 1, 0) - nim
    return cim, nim


def get_ins_list(image_path, desire_margin_list, y_dir):
    xp = image_path
    fn = xp.split("/")[-1].split(FILE_TYPE_EXTENSION)[0]
    img = imageio.imread(xp)
    img = sim_resize(img, (HEIGHT, WIDTH))


    yp_list = glob.glob(f"{y_dir}{fn}_*{FILE_TYPE_EXTENSION}")

    all_inst_data_list = []
    for yp in yp_list:
        msk = imageio.imread(yp)
        # resize mask
        msk = sim_resize(msk, (HEIGHT, WIDTH))

        if len(msk.shape) == 3:
            msk = msk[:, :, 0]
        cim, nim = split_c_n(msk)
        cim = np.where(cim > 0, 255, 0)
        nim = np.where(nim > 0, 255, 0)

        ## crop nucleus
        idxs, idys = np.nonzero(nim)
        n_bbox = [min(idxs), min(idys), max(idxs) + 1, max(idys) + 1]
        idxs, idys = np.nonzero(cim)
        c_bbox = [min(idxs), min(idys), max(idxs) + 1, max(idys) + 1]
        bbox = [
            max(0, n_bbox[0]),
            max(0, n_bbox[1]),
            min(img.shape[0], n_bbox[2]),
            min(img.shape[1], n_bbox[3]),
        ]
        n_img = img[bbox[0] : bbox[2], bbox[1] : bbox[3]]
        n_msk = nim[bbox[0] : bbox[2], bbox[1] : bbox[3]]

        all_scales_data_list = []
        for scale in desire_margin_list:
            dx = round(scale * n_msk.shape[0] / 2)
            dy = round(scale * n_msk.shape[1] / 2)

            dx, dy = int(dx), int(dy)

            snmsk = np.zeros(
                (n_msk.shape[0] + 2 * dx, n_msk.shape[1] + 2 * dy), dtype=np.uint8
            )
            scmsk = np.zeros(
                (n_msk.shape[0] + 2 * dx, n_msk.shape[1] + 2 * dy), dtype=np.uint8
            )
            simg = np.zeros(
                (n_msk.shape[0] + 2 * dx, n_msk.shape[1] + 2 * dy, 3),
                dtype=np.uint8,
            )

            bbox = [
                max(0, n_bbox[0] - dx),
                max(0, n_bbox[1] - dy),
                min(img.shape[0], n_bbox[2] + dx),
                min(img.shape[1], n_bbox[3] + dy),
            ]

            timg = img[bbox[0] : bbox[2], bbox[1] : bbox[3]]
            tnmsk = nim[bbox[0] : bbox[2], bbox[1] : bbox[3]]
            tcmsk = cim[bbox[0] : bbox[2], bbox[1] : bbox[3]]

            shift_x = round((simg.shape[0] - timg.shape[0]) / 2)
            shift_y = round((simg.shape[1] - timg.shape[1]) / 2)
            simg[
                shift_x : timg.shape[0] + shift_x,
                shift_y : timg.shape[1] + shift_y,
                :,
            ] = timg
            snmsk[
                shift_x : tnmsk.shape[0] + shift_x,
                shift_y : tnmsk.shape[1] + shift_y,
            ] = tnmsk
            scmsk[
                shift_x : tcmsk.shape[0] + shift_x,
                shift_y : tcmsk.shape[1] + shift_y,
            ] = tcmsk

            is_cyto_fully_cov = (
                (bbox[0] <= c_bbox[0]) and (bbox[1] <= c_bbox[1])
            ) and ((bbox[2] >= c_bbox[2]) and (bbox[3] >= c_bbox[3]))
            tdata = {
                "scale": scale,
                "bbox": bbox,
                "bbox_hint": "[x_min, y_min, x_max, y_max]",
                "shift": [shift_x, shift_y],
                "simg_size": snmsk.shape,
                "simg": simg,
                "snmsk": snmsk,
                "scmsk": scmsk,
                "is_cyto_fully_cov": is_cyto_fully_cov,
            }
            all_scales_data_list.append(tdata)

        all_inst_data_list.append(all_scales_data_list)

    data = {
        "meta": {
            "image_name": fn,
            "image_path": xp,
            "image_size": img.shape,
            "total_insts": len(all_inst_data_list),
        },
        "data": all_inst_data_list,
    }

    return data


def split_segpc_train_test(X, Y, tr_p=0.5, vl_p=0.2):
    total = len(X)

    tr_vl_idx = int(total * tr_p)
    vl_te_idx = int(total * (tr_p + vl_p))

    X_tr = X[:tr_vl_idx]
    X_vl = X[tr_vl_idx:vl_te_idx]
    X_te = X[vl_te_idx:]

    Y_tr = Y[:tr_vl_idx]
    Y_vl = Y[tr_vl_idx:vl_te_idx]
    Y_te = Y[vl_te_idx:]

    return X_tr, X_vl, X_te, Y_tr, Y_vl, Y_te


@click.command()
@click.argument("data_dir", type=click.Path(exists=True))
def main(data_dir: str):
    INPUT_SIZE = (224, 224)
    GENERATE_NEW_DATA = True
    SCALES = [2.5]

    DATASET_DIR = f"{data_dir}/segpc/TCIA_SegPC_dataset/"
    TRAIN_DIR = DATASET_DIR + "train/"
    VAL_DIR = DATASET_DIR + "validation/"
    SAVE_DATA_IN = f"{data_dir}/segpc/np"
    os.makedirs(SAVE_DATA_IN, exist_ok=True)

    TRAIN_X_DIR = TRAIN_DIR + "x/"
    TRAIN_Y_DIR = TRAIN_DIR + "y/"
    VAL_X_DIR = VAL_DIR + "x/"
    VAL_Y_DIR = VAL_DIR + "y/"

    VISUALIZE_DIR = "../visualize/"

    if GENERATE_NEW_DATA:
        # for train folder
        tr_X_path_list = glob.glob(TRAIN_X_DIR + "*" + FILE_TYPE_EXTENSION)
        tr_X, tr_Y = [], []
        for xp in tqdm(tr_X_path_list, desc="Train"):
            ins_list = get_ins_list(xp, SCALES, TRAIN_Y_DIR)["data"]
            for ins in ins_list:
                for ins_scale in ins:
                    img = resize_pad(ins_scale["simg"], INPUT_SIZE)
                    snmsk = resize_pad(ins_scale["snmsk"], INPUT_SIZE)
                    scmsk = resize_pad(ins_scale["scmsk"], INPUT_SIZE)
                    #                 for i in range(3):
                    #                     img[:,:,i] = cv2.equalizeHist(np.uint8(img[:,:,i]))
                    x = np.concatenate([img, np.expand_dims(snmsk, -1)], -1)
                    tr_X.append(np.uint8(x))
                    tr_Y.append(np.uint8(scmsk + snmsk))

        print(len(tr_X))
        # for test folder
        val_X_path_list = glob.glob(VAL_X_DIR + "*" + FILE_TYPE_EXTENSION)
        val_X, val_Y = [], []
        for xp in tqdm(val_X_path_list, desc="Validation"):
            ins_list = get_ins_list(xp, SCALES, VAL_Y_DIR)["data"]
            for ins in ins_list:
                for ins_scale in ins:
                    img = resize_pad(ins_scale["simg"], INPUT_SIZE)
                    snmsk = resize_pad(ins_scale["snmsk"], INPUT_SIZE)
                    scmsk = resize_pad(ins_scale["scmsk"], INPUT_SIZE)
                    #                 for i in range(3):
                    #                     img[:,:,i] = cv2.equalizeHist(np.uint8(img[:,:,i]))
                    x = np.concatenate([img, np.expand_dims(snmsk, -1)], -1)
                    val_X.append(np.uint8(x))
                    val_Y.append(np.uint8(scmsk + snmsk))

        # split to train and validation
        print("converting list of images to numpy array...")
        X = np.array(tr_X + val_X)
        Y = np.array(tr_Y + val_Y)
        print(f"total-X: {X.shape}, total-Y: {Y.shape}")

        X_tr, X_vl, X_te, Y_tr, Y_vl, Y_te = split_segpc_train_test(
            X, Y, tr_p=0.5, vl_p=0.2
        )
        print(f"tr:{X_tr.shape}, vl:{X_vl.shape}, te:{X_te.shape}")

        numpy_X = X_tr.copy()
        numpy_Y = Y_tr.copy()
        # ===================== finished data augmentation ====================

        # ------------------------- start storing -----------------------
        ADD = SAVE_DATA_IN

        # total
        np.save(f"{ADD}/cyts_{INPUT_SIZE[0]}x{INPUT_SIZE[1]}_s{SCALES[0]}_X", X)
        np.save(f"{ADD}/cyts_{INPUT_SIZE[0]}x{INPUT_SIZE[1]}_s{SCALES[0]}_Y", Y)

        # train
        np.save(f"{ADD}/cyts_tr_{INPUT_SIZE[0]}x{INPUT_SIZE[1]}_s{SCALES[0]}_X", X_tr)
        np.save(f"{ADD}/cyts_tr_{INPUT_SIZE[0]}x{INPUT_SIZE[1]}_s{SCALES[0]}_Y", Y_tr)

        # validation
        np.save(f"{ADD}/cyts_vl_{INPUT_SIZE[0]}x{INPUT_SIZE[1]}_s{SCALES[0]}_X", X_vl)
        np.save(f"{ADD}/cyts_vl_{INPUT_SIZE[0]}x{INPUT_SIZE[1]}_s{SCALES[0]}_Y", Y_vl)

        # test
        np.save(f"{ADD}/cyts_te_{INPUT_SIZE[0]}x{INPUT_SIZE[1]}_s{SCALES[0]}_X", X_te)
        np.save(f"{ADD}/cyts_te_{INPUT_SIZE[0]}x{INPUT_SIZE[1]}_s{SCALES[0]}_Y", Y_te)
        # ===============================================================

    else:
        ADD = SAVE_DATA_IN

        # total
        X = np.load(f"{ADD}/cyts_{INPUT_SIZE[0]}x{INPUT_SIZE[1]}_s{SCALES[0]}_X")
        Y = np.load(f"{ADD}/cyts_{INPUT_SIZE[0]}x{INPUT_SIZE[1]}_s{SCALES[0]}_Y")

        # train
        X_tr = np.load(f"{ADD}/cyts_tr_{INPUT_SIZE[0]}x{INPUT_SIZE[1]}_s{SCALES[0]}_X")
        Y_tr = np.load(f"{ADD}/cyts_tr_{INPUT_SIZE[0]}x{INPUT_SIZE[1]}_s{SCALES[0]}_Y")

        # validation
        X_vl = np.load(f"{ADD}/cyts_vl_{INPUT_SIZE[0]}x{INPUT_SIZE[1]}_s{SCALES[0]}_X")
        Y_vl = np.load(f"{ADD}/cyts_vl_{INPUT_SIZE[0]}x{INPUT_SIZE[1]}_s{SCALES[0]}_Y")

        # test
        X_te = np.load(f"{ADD}/cyts_te_{INPUT_SIZE[0]}x{INPUT_SIZE[1]}_s{SCALES[0]}_X")
        Y_te = np.load(f"{ADD}/cyts_te_{INPUT_SIZE[0]}x{INPUT_SIZE[1]}_s{SCALES[0]}_Y")

        numpy_X = X_tr.copy()
        numpy_Y = Y_tr.copy()



if __name__ == "__main__":
    main()
