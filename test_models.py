import click
import os
from typing import Tuple

import torch
import torchmetrics
from tqdm import tqdm

import models._uctransnet.Config as uct_config
from datasets.isic import ISIC2018DatasetFast
from datasets.segpc import SegPC2021Dataset
from models._missformer.MISSFormer import MISSFormer
from models._resunet.res_unet import ResUnet
# transformer models needs special treatment
from models._transunet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from models._transunet.vit_seg_modeling_c4 import VisionTransformer
from models._uctransnet.UCTransNet import UCTransNet
from models.attunet import AttU_Net
from models.multiresunet import MultiResUnet
from models.unet import UNet
from models.unetpp import NestedUNet
from utils import load_config

from claimed_test_results import results as claimed_results

MODELS = {
    "attunet": AttU_Net,
    "missformer": MISSFormer,
    "multiresunet": MultiResUnet,
    "resunet": ResUnet,
    "transunet": VisionTransformer,
    "uctransnet": UCTransNet,
    "unet": UNet,
    "unetpp": NestedUNet,
}

DATASETS = {"isic": "2018", "segpc": "2021"}


def load_pretrained_model(dataset_name: str, model_name: str, device) -> torch.nn.Module:
    """
    Initialize a model specified by model_name, and load its weights pretrained
    on the dataset specified by dataset_name.

    Args:
        dataset_name: name of the dataset
        model_name: name of the model

    Returns:
        model: the model with pretrained weights
    """
    year = DATASETS[dataset_name]
    config_path = f"configs/{dataset_name}/{dataset_name}{year}_{model_name}.yaml"
    config = load_config(config_path)

    # initialize model
    if model_name == "transunet":
        model = load_transunet(config)
    elif model_name == "uctransnet":
        model = load_uctransnet(config)
    else:
        model = MODELS[model_name](**config["model"]["params"])

    # load weights
    weights_path = f"{config['model']['save_dir']}/best_model_state_dict.pt"
    weights_path = weights_path[6:]  # remove initial '../..' from path
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)

    return model


def load_transunet(config):
    config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
    config_vit.n_classes = config["model"]["params"]["num_classes"]
    config_vit.n_skip = 3
    INPUT_SIZE = config["dataset"]["input_size"]
    if "R50-ViT-B_16".find("R50") != -1:
        config_vit.patches.grid = (int(INPUT_SIZE / 16), int(INPUT_SIZE / 16))

    model = MODELS["transunet"](config_vit, **config["model"]["params"])
    print(config_vit)
    print(model)

    return model


def load_uctransnet(config):
    config_vit = uct_config.get_CTranS_config()
    model = MODELS["uctransnet"](config_vit, **config["model"]["params"])

    return model


def test_set_loader(name: str, data_dir: str) -> torch.utils.data.DataLoader:
    if name == "isic":
        dataset = ISIC2018DatasetFast(mode="te", data_dir=os.path.join(data_dir, "isic2018/np"))
    elif name == "segpc":
        dataset = SegPC2021Dataset(mode="te", data_dir=os.path.join(data_dir, "segpc2021/np"))
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        **{'batch_size': 16, 'shuffle': False, 'num_workers': 4, 'pin_memory': False}
    )

    return dataloader


def test(
        model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, device, task: str
) -> torchmetrics.MetricCollection:

    test_metrics = torchmetrics.MetricCollection(
        [
            torchmetrics.F1Score(task=task),
            torchmetrics.Accuracy(task=task),
            torchmetrics.Dice(task=task),
            torchmetrics.Precision(task=task),
            torchmetrics.Recall(task=task),
            torchmetrics.Specificity(task=task),
            # IoU
            torchmetrics.JaccardIndex(task=task),
        ]
    )

    model.eval()
    with torch.no_grad():
        evaluator = test_metrics.to(device)
        for batch_data in tqdm(data_loader):
            imgs = batch_data["image"]
            msks = batch_data["mask"]

            imgs = imgs.to(device)
            msks = msks.to(device)

            preds = model(imgs)

            # evaluate by metrics
            preds_ = torch.argmax(preds, 1, keepdim=False).float()
            msks_ = torch.argmax(msks, 1, keepdim=False)
            evaluator.update(preds_, msks_)

    results = {k: v.cpu().numpy() for k, v in evaluator.compute().items()}
    return results


@click.command()
@click.argument("data_dir", type=click.Path(exists=True))
def main(data_dir: str):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    translation = {
            'BinaryAccuracy': 'AC',
            'BinaryPrecision': 'PR',
            'BinaryRecall': 'SE',  # wrong?
            'BinarySpecificity': 'SP',
            'Dice': 'Dice',
            'BinaryJaccardIndex': 'IoU',
            }

    for dataset_name in DATASETS.keys():
        dataloader = test_set_loader(dataset_name, data_dir)

        for model_name in MODELS.keys():
            if model_name == "transunet":  # this one isn't working yet
                continue
            model = load_pretrained_model(dataset_name, model_name, device)

            print(f"Computing test metrics of model {model_name} on dataset {dataset_name}")
            task = "binary"
            metrics = test(model, dataloader, device, task)
            reference_metrics = claimed_results[dataset_name][model_name]
            for k, v in metrics.items():
                if k not in translation:
                    continue
                stat_name = translation[k]
                result = metrics[k]
                claimed_result = reference_metrics[stat_name]
                print(f"{stat_name}: {v:.4f} (claimed {claimed_result:.4f}), diff: {v - claimed_result:.4f}")


if __name__ == "__main__":
    main()
