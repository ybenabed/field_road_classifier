import torch.nn as nn
from torch import save as torchsave
import copy
from datetime import datetime
import os
import pandas as pd


def custom_efficient_net(
    efficient_model,
):
    model = copy.deepcopy(efficient_model)
    model.classifier = nn.Sequential(
        efficient_model.classifier[0],
        efficient_model.classifier[1],
        efficient_model.classifier[2],
        efficient_model.classifier[3],
        nn.ReLU(),
        nn.Linear(in_features=1000, out_features=2, bias=True),
        nn.Softmax(),
    )

    return model


def save_model(model, prefix: str, metrics: dict):
    dtime = datetime.now().strftime("%d%m%Y:%H%M")
    model_name = f"{prefix}_{dtime}"
    os.makedirs(f"outputs/{model_name}")
    torchsave(model.state_dict(), f"outputs/{model_name}/model_final.pt")
    if metrics is not None:
        df = pd.DataFrame.from_dict(metrics, orient="index")
        df.to_csv("metrics.csv")
