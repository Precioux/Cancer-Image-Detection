
# Brain Tumor Detection

This repository contains a Jupyter Notebook for detecting brain tumors using machine learning techniques. The project involves preprocessing medical images, training a model, and evaluating its performance on detecting tumors.

## Project Overview

The Brain Tumor Detection project includes the following key components:

- **Data Preparation:** Organizing and preprocessing medical images for model training.
- **Model Training:** Utilizing deep learning techniques to train a model on the preprocessed images.
- **Evaluation:** Assessing the model's performance on a separate test dataset to evaluate its accuracy in detecting brain tumors.

## Installation

To run this project locally, you will need to have Python 3 installed along with the following libraries:

- `numpy`
- `opencv-python`
- `tensorflow` or `keras`
- `matplotlib`
- `sklearn`
- `tqdm`
- `imutils`

You can install these dependencies using pip:

```bash
pip install numpy opencv-python tensorflow matplotlib scikit-learn tqdm imutils
```

## Usage

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/precioux/brain-tumor-detection.git
   cd brain-tumor-detection
   ```

2. **Open the Jupyter Notebook:**

   Launch Jupyter Notebook and open the `brain-tumor-detection.ipynb` file.

   ```bash
   jupyter notebook brain-tumor-detection.ipynb
   ```

3. **Run the Cells:**

   Execute the cells in the notebook sequentially to load the data, preprocess it, train the model, and evaluate its performance.

4. **Evaluating the Model:**

   The notebook includes code for visualizing the results, including sample predictions and model accuracy metrics.

## Data

The project utilizes a dataset of brain MRI images labeled as either "tumor" or "no tumor". The data is organized into training, validation, and test sets to ensure robust model evaluation.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- The project is inspired by the need for efficient and accurate medical diagnosis using AI.
- Special thanks to the creators of the dataset and the authors of the libraries used in this project.
- Dataset: https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection