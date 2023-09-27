import glob
import os

import click
import numpy as np
from torchvision import transforms, utils
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from tqdm import tqdm


@click.command()
@click.argument("data_dir")
def main(data_dir: str):
    INPUT_SIZE = 224

    img_transform = transforms.Compose(
        [
            transforms.Resize(
                size=[INPUT_SIZE, INPUT_SIZE],
                interpolation=transforms.functional.InterpolationMode.BILINEAR,
            ),
        ]
    )
    msk_transform = transforms.Compose(
        [
            transforms.Resize(
                size=[INPUT_SIZE, INPUT_SIZE],
                interpolation=transforms.functional.InterpolationMode.NEAREST,
            ),
        ]
    )

    data_prefix = "ISIC_"
    target_postfix = "_segmentation"
    target_fex = "png"
    input_fex = "jpg"
    data_dir = os.path.join(data_dir, "isic2018")
    imgs_dir = os.path.join(data_dir, "inputs")
    msks_dir = os.path.join(data_dir, "outputs")

    img_dirs = glob.glob(f"{imgs_dir}/*.{input_fex}")
    data_ids = [d.split(data_prefix)[1].split(f".{input_fex}")[0] for d in img_dirs]

    def get_img_by_id(id):
        img_dir = os.path.join(imgs_dir, f"{data_prefix}{id}.{input_fex}")
        img = read_image(img_dir, ImageReadMode.RGB)
        return img

    def get_msk_by_id(id):
        msk_dir = os.path.join(
            msks_dir, f"{data_prefix}{id}{target_postfix}.{target_fex}"
        )
        msk = read_image(msk_dir, ImageReadMode.GRAY)
        return msk

    imgs = []
    msks = []
    for data_id in tqdm(data_ids):
        img = get_img_by_id(data_id)
        msk = get_msk_by_id(data_id)

        if img_transform:
            img = img_transform(img)
            img = (img - img.min()) / (img.max() - img.min())
        if msk_transform:
            msk = msk_transform(msk)
            msk = (msk - msk.min()) / (msk.max() - msk.min())

        img = img.numpy()
        msk = msk.numpy()

        imgs.append(img)
        msks.append(msk)

    X = np.array(imgs)
    Y = np.array(msks)

    os.makedirs(f"{data_dir}/np", exist_ok=True)
    np.save(f"{data_dir}/np/X_tr_224x224", X)
    np.save(f"{data_dir}/np/Y_tr_224x224", Y)


if __name__ == "__main__":
    main()
