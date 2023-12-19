from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torch
from eval import calculate_metrics
from model import custom_efficient_net, save_train_model
from typing import Tuple


class FieldRoadClassifier:
    """
    A classifier for distinguishing between field and road images using EfficientNet.

    Attributes:
    - target_size (tuple): Target size for image resizing (default: (256, 256)).
    - device (torch.device): Device to be used for computation ('cuda' or 'cpu').
    - model (torch.nn.Module): Custom model for classification.
    - prefix (str): Prefix for saving the trained model.

    """

    def __init__(self, prefix: str, target_size: Tuple[int, int] = (256, 256)):
        """
        Initialize the FieldRoadClassifier.

        Parameters:
        - prefix (str): Prefix for saving the trained model.
        - target_size (Tuple[int, int]): Target size for image resizing (default: (256, 256)).
        """
        self.target_size = target_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.prefix = prefix

    def load_data(
        self, dataset_path: str
    ) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
        """
        Load and prepare the dataset for training and validation.

        Parameters:
        - dataset_path (str): Path to the root of the dataset.

        Returns:
        - Tuple[DataLoader, DataLoader, torch.Tensor]: Tuple containing train DataLoader, validation DataLoader, and class weights.
        """
        transform = transforms.Compose(
            [
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
            ]
        )

        dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
        self.class_labels = dataset.classes
        class_weights = self.ds_class_weights(dataset)

        train_size = max(len(dataset) - 150, int(0.8 * len(dataset)))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        return self.train_loader, self.val_loader, class_weights

    def ds_class_weights(self, dataset: datasets.ImageFolder) -> torch.Tensor:
        """
        Calculate class weights for the dataset.

        Parameters:
        - dataset (datasets.ImageFolder): Image dataset.

        Returns:
        - torch.Tensor: Class weights.
        """
        class_counts = torch.tensor(dataset.targets).bincount()
        total_samples = len(dataset)
        num_classes = len(class_counts)
        class_weights = total_samples / (num_classes * class_counts.float())
        return class_weights

    def load_model(self) -> torch.nn.Module:
        """
        Load a pre-trained EfficientNet model and create a custom model based on it.

        Returns:
        - torch.nn.Module: Custom model.
        """
        efficient_model = torch.hub.load(
            "NVIDIA/DeepLearningExamples:torchhub",
            "nvidia_efficientnet_b0",
            pretrained=True,
        )
        model = custom_efficient_net(efficient_model)
        return model

    def fit(self, dataset_path: str, num_epochs: int = 10, VAL_TOLERANCE: int = 1):
        """
        Train the model on the specified dataset.

        Parameters:
        - dataset_path (str): Path to the root of the dataset.
        - num_epochs (int): Number of training epochs (default: 10).
        - VAL_TOLERANCE (int): Tolerance for early stopping (default: 1).
        """
        train_loader, val_loader, _ = self.load_data(dataset_path=dataset_path)
        self.model = self.model.to(self.device)
        optimizer = torch.optim.Adagrad(self.model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        self.training(
            num_epochs,
            VAL_TOLERANCE,
            train_loader,
            val_loader,
            self.model,
            optimizer,
            criterion,
        )

    def training(
        self,
        num_epochs: int,
        VAL_TOLERANCE: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
    ):
        """
        Train the model for the specified number of epochs with early stopping.

        Parameters:
        - num_epochs (int): Number of training epochs.
        - VAL_TOLERANCE (int): Tolerance for early stopping.
        - train_loader (DataLoader): DataLoader for training set.
        - val_loader (DataLoader): DataLoader for validation set.
        - model (torch.nn.Module): Model to be trained.
        - optimizer (torch.optim.Optimizer): Model optimizer.
        - criterion (torch.nn.Module): Loss function.
        """
        metrics = {}
        val_counter = 0
        min_validation = float("inf")

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            val_loss = 0

            conf_matrix = {
                "True Positives": 0,
                "False Positives": 0,
                "True Negatives": 0,
                "False Negatives": 0,
            }

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                loss = self.train_step(model, optimizer, criterion, inputs, labels)
                epoch_loss += loss.item()

            model.eval()

            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                loss, confusion_matrix = self.val_step(model, criterion, inputs, labels)
                conf_matrix["True Positives"] += confusion_matrix["True Positives"]
                conf_matrix["False Positives"] += confusion_matrix["False Positives"]
                conf_matrix["True Negatives"] += confusion_matrix["True Negatives"]
                conf_matrix["False Negatives"] += confusion_matrix["False Negatives"]
                val_loss += loss.item()

            metrics[f"Epoch {epoch+1}"] = {
                "Total Loss": epoch_loss,
                "Val Loss": val_loss,
            }
            if val_loss >= min_validation:
                val_counter += 1
                if val_counter >= VAL_TOLERANCE:
                    print(
                        f"Epoch {epoch+1}/{num_epochs} Total loss: {epoch_loss}\t Validation loss: {val_loss}"
                    )
                    print("Early Stopping Triggered")
                    break
            else:
                val_counter = 0
                min_validation = val_loss

            print(
                f"Epoch {epoch+1}/{num_epochs} Total loss: {epoch_loss}\t Validation loss: {val_loss}"
            )

        model_folder = save_train_model(
            model=model, prefix=self.prefix, metrics=metrics, conf_matrix=conf_matrix
        )
        self.output_folder = model_folder

    def train_step(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform a single training step.

        Parameters:
        - model (torch.nn.Module): Model to be trained.
        - optimizer (torch.optim.Optimizer): Model optimizer.
        - criterion (torch.nn.Module): Loss function.
        - inputs (torch.Tensor): Input data.
        - labels (torch.Tensor): Ground truth labels.

        Returns:
        - torch.Tensor: Loss value.
        """
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        return loss

    def val_step(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Perform a single validation step.

        Parameters:
        - model (torch.nn.Module): Model to be validated.
        - criterion (torch.nn.Module): Loss function.
        - inputs (torch.Tensor): Input data.
        - labels (torch.Tensor): Ground truth labels.

        Returns:
        - Tuple[torch.Tensor, dict]: Tuple containing loss value and evaluation metrics.
        """
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        metrics = calculate_metrics(outputs=outputs, labels=labels)
        return loss, metrics

    def evaluation(self, model: torch.nn.Module, val_loader: DataLoader):
        """
        Evaluate the model on the validation set.

        Parameters:
        - model (torch.nn.Module): Model to be evaluated.
        - val_loader (DataLoader): DataLoader for validation set.
        """
        model.eval()
        eval_labels, eval_outs = [], []
        for inputs, labels in val_loader:
            eval_labels.append(eval_labels)
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = model(inputs)
            eval_outs.append(outputs)

        eval_labels = torch.cat(eval_labels)
        eval_outs = torch.cat(eval_outs)

        calculate_metrics(eval_outs, eval_labels)
