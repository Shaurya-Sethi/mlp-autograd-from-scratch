
# MLP & Autograd From Scratch (Vectorized Tensors)

This repository contains a minimal yet powerful implementation of a Multi-Layer Perceptron (MLP) and autograd engine built entirely from scratch in Python, without relying on any external deep learning frameworks. The project showcases the full pipeline for training a neural network on tabular data, using custom vectorized tensor operations and automatic differentiation.

## Inspiration

This project is inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) series on YouTube, where he builds an autograd engine using only scalar values. After following his lectures and reimplementing micrograd for my own learning, I extended the core ideas to operate on **vectorized tensors** instead of scalars. This extension allows efficient training of deep models on real-world datasets and serves as an educational stepping stone towards understanding frameworks like PyTorch and TensorFlow.

## Features

* Custom autograd engine supporting vectorized tensor operations (addition, multiplication, matrix multiplication, etc.)
* Core neural network building blocks: Linear, BatchNorm, SELU activation, Dropout
* Modern training utilities: AdamW optimizer, learning rate scheduler, early stopping
* Fully vectorized MLP architecture for multi-class classification
* Reproducible results on tabular data
* No external ML frameworks (only NumPy and scikit-learn for preprocessing)

## Getting Started

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/mlp-autograd-from-scratch.git
   cd mlp-autograd-from-scratch
   ```
2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The code expects a tabular CSV file with features and labels. A small sample (`data/example.csv`) is included for demonstration purposes.

### Quick Steps to Use Your Own Dataset

* Your CSV file **must include columns matching those expected by the data loading functions in `src/utils.py`**. See `validate_dataframe()` for the exact column list.
* If your dataset uses different column names or has different features, **edit the functions in `src/utils.py` accordingly**.
* The code does **not** require any specific dataset—**any tabular dataset for multi-class classification can be used**, as long as the preprocessing and columns match.
* **Example: To use your own data**,

  1. Place your file in the `data/` folder (or anywhere you like).
  2. Run the script:
     `python train.py --csv data/yourfile.csv`
  3. If columns differ, modify the data loading and validation in `src/utils.py`.

The code expects a tabular CSV file with features and labels. A small sample (`data/example.csv`) is included for demonstration purposes. To use your own dataset:

* Ensure your CSV contains columns that match those expected by the data loading utilities in `utils.py`.
* If your dataset has different features or column names, you must adjust `load_and_preprocess_data()` and `validate_dataframe()` in `utils.py` to fit your data.

### Training

Run the training script with the provided sample dataset or your own:

```bash
python train.py --csv data/example.csv
```

Or specify your custom dataset path:

```bash
python train.py --csv /path/to/your/dataset.csv
```

### Customization

* **To adapt for custom datasets:**

  * Change the path passed to the training script.
  * Update the column handling in `utils.py` to match your feature and label columns.
* **To modify the architecture:**

  * Edit `mlp.py` to add/remove layers, change activation functions, or adjust hidden sizes.

## File Structure

* `tensor.py`      : Core Tensor class with vectorized autograd
* `layers.py`      : Linear, BatchNorm1d, SELU, Dropout implementations
* `mlp.py`         : MLP model definition
* `optimizer.py`   : AdamW optimizer, learning rate scheduler
* `losses.py`      : Cross-entropy and related loss functions
* `utils.py`       : Data loading, preprocessing, save/load utilities
* `train.py`       : Main training and evaluation script
* `requirements.txt`: Package requirements
* `data/`          : Folder for example/sample datasets

## Acknowledgements

Special thanks to Andrej Karpathy for his micrograd project and YouTube lectures, which provided the foundation and motivation for this implementation.

## License

This repository is released under the MIT License.