# VLGanimalclassification

## Google Drive Link of the dataset I've used(also includes the unseen classes data I collected) 
https://drive.google.com/drive/folders/1vVCL4XXPMRaDsoIUTBdhcZvSMv_XTgH9?usp=sharing

## Image Classification using TensorFlow

This project involves building and utilizing a Convolutional Neural Network (CNN) for image classification using TensorFlow. Below are the key details and explanations of the implementation.

## Features
- **Framework**: TensorFlow and Keras for model development and training.
- **Preprocessing**: Image resizing, augmentation, and normalization.
- **Model Architecture**: EfficientNetV2L(last 80 layers unfreezed), a powerful CNN for image classification with 2 layers and a dropout layer..
- **Training**: Configurable epochs, batch sizes, and optimizers.
- **Inference**: Image classification using the trained CNN model.

## Files and Directory Structure
- **Model.ipynb**: The main implementation notebook for training and testing the CNN.
- **Dataset**: The dataset contains a train folder with 40 classes, each with around 250 images, and 10 additional unseen classes with approximately 125 images per class. The validation dataset is derived from the train dataset.

## Implementation Steps

### 1. Importing Libraries
The code uses TensorFlow, Keras, NumPy, and Matplotlib for building, training, and visualizing the model's performance.

### 2. Data Preprocessing
- **Resizing**: Images are resized to a uniform size (240x240 pixels).
- **Augmentation**: Data augmentation techniques (like rotation, flipping, etc.) are applied to increase dataset variability.

### 3. CNN Architecture
The model is defined using the Keras Sequential API with layers:
- **EfficientNetV2L (last 80 layers unfreezed)**: A pre-trained model used as the backbone for feature extraction and classification.
- **Fully Connected Layers**: Perform classification based on extracted features.
- **Dropout**: Added to prevent overfitting.

### 4. Training Configuration
- **Optimizer**: Configured  SGD.
- **Loss Function**: SparseCategoricalCrossentropy(from_logits=True) for multi-class classification.
- **Metrics**: Accuracy is tracked during training.

### 5. Model Evaluation
- The trained model is evaluated using a validation set and metrics like accuracy and loss.

### 6. Inference Function
```python
def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(240,240))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = animal_names[np.argmax(result)]
    return outcome
```
This function:
- Preprocesses the input image.
- Feeds it into the trained model.
- Returns the predicted class name based on the model's output probabilities.

## Usage
1. Place the dataset in the appropriate directory structure.
2. Train the model using the notebook.
3. Save the trained model for future inference.
4. Use the `classify_images` function to predict classes for new images.

## Vital Part of Implementation
- The function `classify_images` demonstrates how to preprocess an image and use the trained model to make predictions.
- The CNN architecture balances complexity and efficiency for robust image classification.

## Dependencies
- TensorFlow
- NumPy
- Matplotlib

## Note
Ensure your dataset is well-prepared and balanced to achieve optimal results during training.

