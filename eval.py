import torch
from sklearn.metrics import confusion_matrix
from typing import Dict, Union


def calculate_metrics(
    outputs: torch.Tensor, labels: torch.Tensor
) -> Dict[str, Union[int, float]]:
    """
    Calculate various classification metrics based on softmax outputs and true labels.

    Parameters:
    - outputs (torch.Tensor): The softmax outputs from the model.
    - labels (torch.Tensor): The true labels for each example (0 or 1).

    Returns:
    - Dict[str, Union[int, float]]: A dictionary containing the following metrics:
        - 'True Positives': Number of true positives.
        - 'False Positives': Number of false positives.
        - 'True Negatives': Number of true negatives.
        - 'False Negatives': Number of false negatives.
        - 'Precision': Precision score.
        - 'Recall': Recall score.
        - 'F1 Score': F1 score.
    """

    _, predicted_labels = torch.max(outputs, 1)

    conf_matrix = confusion_matrix(labels.cpu().numpy(), predicted_labels.cpu().numpy())

    tn, fp, fn, tp = conf_matrix.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "True Positives": tp,
        "False Positives": fp,
        "True Negatives": tn,
        "False Negatives": fn,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score,
    }
