# Brain Tumor Classification with a Convolutional Neural Network

## Overview
This project demonstrates how to build a convolutional neural network (CNN) in TensorFlow & Keras for detecting brain tumors from MRI images. The dataset contains MRI scans labeled as _tumorous_ or _non-tumorous_ and was sourced from Kaggle. 

## About the Data
- **Original dataset size**: 253 images
  - 155 _tumorous_ (positive)
  - 98 _non-tumorous_ (negative)
- After augmentation, the dataset increased to **2065 images**:
  - 1085 _tumorous_ (positive)
  - 980 _non-tumorous_ (negative)

> **Note**: The augmented dataset includes the original images as well. All images (original and augmented) are in a folder named `augmented data`.

## Data Augmentation
Because the original dataset was relatively small, data augmentation was used to:
1. Increase the number of training examples.
2. Reduce the data imbalance between the two classes.
3. Help the model generalize better and mitigate overfitting.

## Data Preprocessing
For each image, these preprocessing steps were applied:
1. **Cropping**: Remove the unimportant regions to focus on the brain.
2. **Resizing**: Convert each image to a uniform shape of `(240, 240, 3)`.
3. **Normalization**: Scale pixel values to the range `[0, 1]`.

## Data Split
The dataset was divided as follows:
- **70%** for training
- **15%** for validation
- **15%** for testing

## Neural Network Architecture
Below is the simplified architecture used in this project:

1. **Zero Padding** layer
2. **Convolutional** layer: 32 filters, each `(7 × 7)`, stride = 1
3. **Batch Normalization** layer
4. **ReLU Activation** layer
5. **Max Pooling** layer: `(4 × 4)`, stride = 4
6. **Max Pooling** layer: `(4 × 4)`, stride = 4
7. **Flatten** layer
8. **Dense** layer with a single neuron + **Sigmoid** activation (binary classification)

### Why This Architecture?
- **Simplicity**: Larger networks like ResNet50 or VGG16 easily overfit this small dataset and can be computationally intensive.
- **Efficiency**: A smaller, custom CNN is faster to train and works well with limited hardware resources.
- **Effectiveness**: Despite its simplicity, it achieves competitive performance on this dataset.

## Training the Model
- Trained for **24 epochs**.
- The best validation accuracy was reached at the **23rd** epoch.

**Loss and Accuracy Plots**  
<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/03ea9e21-a543-449a-9c56-388abe4208de" alt="Accuracy" width="400"></td>
    <td><img src="https://github.com/user-attachments/assets/32855b05-84c0-4c77-b118-6d3cfd58d2eb" alt="Loss" width="400"></td>
  </tr>
</table>

## Results
Using the best-performing model (saved at the epoch with the highest validation accuracy), we observe the following on the **test set**:
- **Accuracy**: 88.7%
- **F1 Score**: 0.88

### Performance Table

|            | Validation Set | Test Set |
|------------|---------------:|---------:|
| **Accuracy** | 91%           | 89%      |
| **F1 Score** | 0.91          | 0.88     |

## Final Notes
1. The IPython notebooks contain all the source code for the data preprocessing, model building, and training steps.
2. Model weights are provided. For example, to load the best model:
   ```python
   from tensorflow.keras.models import load_model
   best_model = load_model(filepath='models/cnn-parameters-improvement-23-0.91.model')
