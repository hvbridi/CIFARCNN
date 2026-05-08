# PyTorch Convolutional Model for CIFAR-100 Built from Scratch

## CIFAR-100 is a machine learning dataset composed of 60000 low-resolution images split into 100 different classes
## This project focused on implementing a Convolutional Neural Network from scratch instead of loading pre-made ones in order to further improve my data handling and pipeline creation skills

### Architecture choices
- The model utilizes `Conv2d`, `BatchNorm2d` and `MaxPool2d` in order to extract features from the images, increasing the complexity of the neurons in each layer to be able to comprehend more details of the images
- `Dropout` (which increased gradually per each layer), data augmentation and BatchNorm2d were used to avoid overfitting and ensure a robust model that can reliably predict the test data
- Two `Linear` transformations were made in the last layer in order to normalize the data gradually instead of directly making the final neurons

### Hyperparameters
- **Optimizer:** `Adam` with a initial 0.0005 learning rate
- **Scheduler:** `StepLr` with a 0.2 gamma each 20 epochs
- **Loss Function:** `CrossEntropyLoss`

### Data Augmentation
To improve generalization, the following transforms were applied:
1. `RandomHorizontalFlip`
2. `RandomRotation`
3. `RandomCrop`

### Performance
- **Final Training Accuracy:** 58%
- **Final Test Accuracy:** 57%
- **Training Time:** 50 Epochs in ~18 minutes on CUDA

## Sample prediction
The jupyter notebook includes a inference script that loads the model_weights.pth to predict random test images

### Prerequisites
- Python 3.x
- PyTorch
- Torchvision
- Matplotlib
- Pillow (PIL)

### Usage
1. Clone the repository.
2. Run the training script:
   `python your_script_name.py`
3. The model will automatically download the CIFAR-100 dataset to the `/data` folder, train for 50 epochs, and save the weights.