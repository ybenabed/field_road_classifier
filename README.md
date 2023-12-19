# Field Road Image Classification Repository

Welcome to the Field Road Image Classification repository! This project focuses on classifying images into two categories: field and road. We leverage a pre-trained EfficientNet model, fine-tuned and customized for our specific task. Data augmentation techniques are employed to enhance the training dataset and improve model generalization.

## Repository Structure

- [data](data): Contains both the original and augmented datasets, as well as test images.
- [outputs](outputs): Holds the outputs of the training process. Each model has its dedicated folder, including the `.pt` file, `metrics.csv` (containing training and validation loss), and `confusion_matrix.csv` for the latest trained models.
- [data_augmentation.py](./data_augmentation.py): This script provides functions for data augmentation.
- [demo.ipynb](./demo.ipynb): A Jupyter notebook that illustrates various training operations with visualizations and explanations.
- [eval.py](./eval.py): Functions for computing confusion matrices, recall, and precision from model outputs and labels.
- [model.py](./model.py): Includes functionality for downloading and customizing the EfficientNet model.
- [train.py](./train.py): The script for fine-tuning the model. It features the important `FieldRoadClassifier` class.


## Repository Structure

To get started with this repository, follow these steps:

1. Clone the repository:
```
git clone https://github.com/ybenabed/field_road_classifier.git
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Explore the `demo.ipynb` notebook for an interactive walkthrough of the training process.

4. Use the `train.py` script to fine-tune the model on your dataset:

```
python train.py --prefix output_prefix --path_to_data path\to\dataset --val_tolerance 2
```