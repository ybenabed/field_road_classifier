import torch.nn as nn
from torch import save as torchsave
import copy
from datetime import datetime


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


def save_model(model, prefix):
    dtime = datetime.now().strftime("%d%m%Y:%H%M")
    model_name = f"{prefix}_{dtime}.pt"
    torchsave(model.state_dict(), model_name)
