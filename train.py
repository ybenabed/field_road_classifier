from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import torch
from model import custom_efficient_net, save_train_model


class FieldRoadClassifier:
    """
    A classifier for distinguishing between field and road images using EfficientNet.

    Attributes:
    - target_size (tuple): Target size for image resizing (default: (256, 256)).
    - device (torch.device): Device to be used for computation ('cuda' or 'cpu').
    - model (torch.nn.Module): Custom model for classification.

    Methods:
    - load_data(dataset_path: str) -> tuple: Load and prepare the dataset for training and validation.
    - load_model() -> torch.nn.Module: Load a pre-trained EfficientNet model and create a custom model.
    - fit(dataset_path: str, num_epochs: int = 10, VAL_TOLERANCE: int = 1): Train the model on the specified dataset.
    - training(num_epochs: int, VAL_TOLERANCE: int, train_loader: DataLoader, val_loader: DataLoader,
               model: torch.nn.Module, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module):
               Train the model for the specified number of epochs with early stopping.
    - train_step(model, optimizer, criterion, inputs, labels): Perform a single training step.
    - val_step(model, criterion, inputs, labels): Perform a single validation step.
    """

    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()

    def load_data(self, dataset_path: str) -> tuple:
        """
        Load and prepare the dataset for training and validation.

        Parameters:
        - dataset_path (str): Path to the root of the dataset.

        Returns:
        - tuple: Tuple containing train DataLoader and validation DataLoader.
        """
        transform = transforms.Compose(
            [
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
            ]
        )

        dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

        train_size = max(len(dataset) - 150, int(0.8 * len(dataset)))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        return train_loader, val_loader

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
        train_loader, val_loader = self.load_data(dataset_path=dataset_path)
        self.model = self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
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
        metrics = {}
        val_counter = 0
        min_validation = float("inf")

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            val_loss = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                loss = self.train_step(model, optimizer, criterion, inputs, labels)
                epoch_loss += loss.item()

            model.eval()
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                loss = self.val_step(model, criterion, inputs, labels)
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
            model=model, prefix="wout_augmentation", metrics=metrics
        )
        self.output_folder = model_folder

    def train_step(self, model, optimizer, criterion, inputs, labels):
        """
        Perform a single training step.
        """

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        return loss

    def val_step(self, model, criterion, inputs, labels):
        """
        Perform a single validation step.
        """

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        return loss
