import torch.nn as nn
from torch import save as torchsave, load as torchload
import copy
from datetime import datetime
import os
import json
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


def save_train_model(model, prefix: str, metrics: dict, conf_matrix: dict) -> str:
    """
    Save the PyTorch model, metrics, and timestamped model information.

    Parameters:
    - model: The PyTorch model to be saved.
    - prefix (str): A prefix for the model name.
    - metrics (dict): Dictionary containing training metrics.

    The saved files include:
    - A folder in the "outputs" directory with a timestamped model name.
    - The PyTorch model's state dictionary saved as "model_final.pt" in the created folder.
    - If metrics are provided, a "metrics.csv" file saved in the created folder.

    """
    dtime = datetime.now().strftime("%d%m%Y:%H%M")
    model_name = f"{prefix}_{dtime}"
    os.makedirs(f"outputs/{model_name}")
    torchsave(model.state_dict(), f"outputs/{model_name}/model_final.pt")
    if metrics is not None:
        df = pd.DataFrame.from_dict(metrics, orient="index")
        df.to_csv(f"outputs/{model_name}/metrics.csv")

    tn, fp, fn, tp = (
        conf_matrix["True Negatives"],
        conf_matrix["False Positives"],
        conf_matrix["False Negatives"],
        conf_matrix["True Positives"],
    )

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    conf_matrix["Precision"] = precision
    conf_matrix["Recall"] = recall
    conf_matrix["F1 Score"] = f1_score

    metrics_df = pd.DataFrame([conf_matrix])
    metrics_df.to_csv(f"outputs/{model_name}/confusion_matrix.csv", index=False)

    return f"outputs/{model_name}"


def load_model_weights(model, checkpoint_path):
    """
    Load the weights of a PyTorch model from a checkpoint file.

    Parameters:
    - model: PyTorch model
    - checkpoint_path: Path to the checkpoint file containing saved weights

    Returns:
    - model: PyTorch model with loaded weights
    """
    checkpoint = torchload(checkpoint_path)
    model.load_state_dict(checkpoint)

    return model
