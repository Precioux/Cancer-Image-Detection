
# Cancer Image Detection using Deep Learning

This project implements a deep learning model to detect metastatic cancer in medical image patches. The model is trained to analyze small image patches derived from larger digital pathology scans, helping in the detection of cancerous cells.

![Cancer Cell Image](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-GPXX0W5QEN/images/cancer%20cell.jpeg)

## Project Overview

The main goal of this project is to build a cancer image detection algorithm using a convolutional neural network (CNN) to process histopathology images and classify whether they contain metastatic tissue.

### Key Features:
- Use of deep learning with PyTorch to process medical images.
- GPU compatibility check and usage for faster model training.
- Extensive data preprocessing, augmentation, and training pipeline setup.

## Setup

### Prerequisites

Before running the project, ensure that you have the following installed:
- Python 3.x
- PyTorch and related libraries
- Additional dependencies listed in the `requirements.txt` (optional)

### Installing Required Libraries

You can install the necessary libraries by running the following command:

```bash
pip install -r requirements.txt
```

Alternatively, for specific libraries (like PyTorch), run:

```bash
pip install torch torchvision
```

### GPU Support

The model takes advantage of GPU acceleration if a compatible GPU is available. The script checks for GPU availability:

```python
from torch.cuda import is_available, get_device_name

if is_available():
    print(f"The environment has a compatible GPU ({get_device_name()}) available.")
else:
    print("The environment does NOT have a compatible GPU model available.")
```

## Dataset

The dataset used for training and evaluation consists of histopathology images. You will need to prepare your dataset in the required format, ensuring that the images are correctly labeled for cancerous and non-cancerous tissue.

### Preprocessing

The dataset is preprocessed using various image augmentation techniques to improve the generalization of the model. You can adjust augmentation strategies in the notebook.

## Model

The model used is a convolutional neural network (CNN) implemented in PyTorch. It is designed to:
- Extract features from input medical images.
- Classify the images into different categories based on the presence of metastatic tissue.

You can modify or extend the architecture based on specific requirements or add transfer learning if needed.

## Training

To train the model, you can run the training section in the notebook, which includes:

1. Loading the dataset.
2. Defining the CNN model architecture.
3. Training the model on the dataset with specified hyperparameters.

## Results

The notebook includes evaluation metrics such as accuracy, precision, recall, and the confusion matrix to measure the model’s performance on the validation set.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/Precioux/Cancer-Image-Detection.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the notebook to preprocess the dataset, train the model, and evaluate the results.

## Future Improvements

- Experiment with deeper CNN architectures.
- Fine-tune the model using transfer learning.
- Expand the dataset to improve model accuracy.
- Deploy the model in a web application for real-time cancer detection.

## Contributing

Feel free to open issues or pull requests for any improvements or bug fixes.

## License

This project is licensed under the MIT License.
