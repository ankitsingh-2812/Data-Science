# Data-Science
Traffic Sign Clasification

Introduction
Traffic sign classification is a critical component of autonomous driving systems, aiding in the detection and recognition of traffic signs to ensure safe and efficient navigation. This project aims to develop a machine learning model capable of accurately classifying traffic signs from images.

Dataset
The German Traffic Sign Recognition Benchmark (GTSRB) dataset was used, consisting of over 50,000 images of 43 different traffic sign classes. Each image is labeled with the correct traffic sign category.

Methodology
Data Preprocessing
Resizing: Images were resized to a standard dimension of 64x64 pixels.
Normalization: Pixel values were scaled to the range [0, 1].
Augmentation: Techniques such as rotation, zooming, and flipping were applied to increase dataset variability and improve model robustness.
Model Selection
A Convolutional Neural Network (CNN) was chosen due to its effectiveness in image classification tasks. The model architecture includes multiple convolutional layers, max-pooling layers, and fully connected layers.

Model Training
Optimizer: Adam optimizer was used for efficient training.
Loss Function: Categorical cross-entropy was selected as the loss function.
Training: The model was trained using the training dataset, with a validation set used to monitor performance and adjust hyperparameters.
Results
The model achieved a high accuracy on the validation set, demonstrating its capability to classify traffic signs effectively. The performance was evaluated using metrics such as accuracy, precision, recall, and the confusion matrix.

Conclusion
The project successfully developed a robust traffic sign classification model. Future work includes further tuning of hyperparameters, expanding the dataset, and real-time deployment for practical applications in autonomous vehicles.
