# Image Classification Model - Training and Evaluation

## Introduction
This project implements a feedforward neural network for image classification using the Caltech 256 dataset. The model is trained with different hyperparameter configurations to determine the best performing setup.

## Requirements
Ensure you have the following dependencies installed:

```sh
pip install torch torchvision matplotlib numpy
```

## Dataset Preparation
Download and extract the Caltech 256 dataset. Place the dataset in the appropriate directory:

```
path_to_caltech256/  # Training data
path_to_caltech256_test/  # Testing data
```

## Training the Model
The training process consists of:
1. Loading and transforming the dataset.
2. Initializing the neural network with different hyperparameters.
3. Training the model using different optimizers.
4. Selecting the best model based on validation accuracy.

To train the model, run:
```sh
python train.py
```

## Evaluating the Model
After training, the best model is evaluated on the test dataset. The evaluation process involves:
1. Loading the best performing model.
2. Running inference on the test dataset.
3. Computing the final test accuracy.

The best configuration and its corresponding test accuracy are displayed after evaluation.

## Customizing Training Configurations
To experiment with different configurations, modify the `configurations` list in the script:
```python
configurations = [
    {"epochs": 5, "hidden_layers": 3, "hidden_units": 64, "learning_rate": 1e-3, "optimizer_type": 'adam', "batch_size": 32, "activation_fn": nn.ReLU},
    {"epochs": 10, "hidden_layers": 4, "hidden_units": 128, "learning_rate": 1e-4, "optimizer_type": 'sgd', "batch_size": 64, "activation_fn": nn.ReLU},
    # Add more configurations as needed
]
```

## Issues & Troubleshooting
- Ensure the dataset path is correctly set in the script.
- If CUDA is available, use GPU acceleration by modifying the code to:
  ```python
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  ```
- For memory issues, reduce the batch size.

## Conclusion
This script helps train and evaluate different model configurations to find the best performing model for image classification using the Caltech 256 dataset.

