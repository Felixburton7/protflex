# 🧬 ProtFlex: Deep Learning for Protein Flexibility Prediction 🧪

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-red.svg)](https://pytorch.org/)

**ProtFlex** is a deep learning framework that predicts protein flexibility (RMSF - Root Mean Square Fluctuation) from voxelized protein structures using advanced 3D convolutional neural networks.

## 📋 Table of Contents

- [Overview](#-overview)
- [Installation](#-installation)
- [Test Run Instructions](#-test-run-instructions)
- [Workflow Diagram](#-workflow-diagram)
- [Common Errors & Fixes](#-common-errors--fixes)
- [Advanced Usage](#-advanced-usage)

## 🔍 Overview

ProtFlex analyzes protein structure data to predict residue flexibility, which is crucial for understanding protein function and dynamics. The system uses 3D CNNs to process voxelized protein structures and outputs RMSF values for each residue.

## 🛠 Installation

### Prerequisites

- Python 3.7+ 
- CUDA-compatible GPU (optional but recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/username/protflex.git
cd protflex
```

### Step 2: Create a Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install the required packages
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn h5py pyyaml tqdm
```

### Step 4: Install ProtFlex

```bash
# Install in development mode
pip install -e .
```

### Step 5: Verify Installation

```bash
# Check if the CLI is properly installed
python -m protflex --help
```

You should see the help message for the ProtFlex CLI.

## 🚀 Test Run Instructions

### Step 1: Prepare a Configuration File

Create a custom configuration file or use the default one. The default configuration file is located at `config/config.yaml`.

```bash
# Copy the default configuration
cp config/config.yaml my_config.yaml
```

### Step 2: Modify the Configuration

Edit `my_config.yaml` to set the correct paths for your data and output directories:

```yaml
input:
  data_dir: /path/to/your/data        # Update this path
  voxel_dir: processed/voxelized_output
  rmsf_dir: interim/per-residue-rmsf
  temperature: 320  # Can also be "average"
  domain_ids: []    # Empty means process all domains

output:
  base_dir: /path/to/results           # Update this path
  model_dir: models
  results_dir: results
  visualizations_dir: visualizations
  log_file: protflex.log
```

⚠️ **Important**: Make sure the paths exist and you have read/write permissions.

### Step 3: Prepare Test Data

For a test run, you'll need:
1. Voxelized protein structure files (HDF5 format)
2. RMSF data files (CSV format)

Ensure your data follows this structure:
```
data_dir/
├── processed/voxelized_output/
│   └── domain_id/
│       └── 320/                      # Temperature directory
│           └── residue_XXX.hdf5      # Voxelized structures
└── interim/per-residue-rmsf/
    └── 320/                          # Temperature directory
        └── domain_id_temperature_320_average_rmsf.csv
```

### Step 4: Run Training

```bash
python -m protflex train --config my_config.yaml \
    --model multipath_rmsf_net \
    --batch-size 16 \
    --epochs 50 \
    --learning-rate 0.001
```

### Step 5: Evaluate the Model

```bash
python -m protflex evaluate \
    --model-path ~/protflex_results/models/best_model.pth \
    --config my_config.yaml
```

### Step 6: Make Predictions

```bash
python -m protflex predict \
    --model-path ~/protflex_results/models/best_model.pth \
    --config my_config.yaml \
    --plot
```

### Step 7: View Results

Check the output directory for:
- Trained models in `model_dir`
- Evaluation metrics in `results_dir/evaluation`
- Predictions in `results_dir/predictions`
- Visualizations in `visualizations_dir`

## 🔄 Workflow Diagram

The workflow of ProtFlex involves several key steps:

1. **Data Preparation**: Voxelized protein structures and RMSF values
2. **Configuration**: Set up parameters for the pipeline
3. **Model Training**: Train 3D CNN models on the prepared data
4. **Evaluation**: Assess model performance
5. **Prediction**: Generate RMSF predictions for new structures
6. **Visualization**: Analyze results with visualizations

## ⚠️ Common Errors & Fixes

### 1. Path Configuration Issues
**Error**: `FileNotFoundError: No such file or directory`

**Fix**: 
- Ensure all paths in `config.yaml` exist
- Use absolute paths instead of relative paths
- Check file permissions

```yaml
# Example correct path configuration
input:
  data_dir: /absolute/path/to/data
output:
  base_dir: /absolute/path/to/output
```

### 2. Data Format Errors
**Error**: `KeyError: 'inputs'` or `Error reading voxel file`

**Fix**:
- Ensure HDF5 files follow the expected structure
- Check that your voxel files contain an 'inputs' dataset or 'voxel_data' dataset
- Verify that RMSF CSV files have the correct format with 'resid' and RMSF columns

### 3. GPU Memory Issues
**Error**: `RuntimeError: CUDA out of memory`

**Fix**:
- Reduce batch size in your configuration
- Use a smaller model architecture
- Force CPU usage with the `--cpu` flag if GPU memory is limited

```bash
python -m protflex train --config my_config.yaml --cpu
```

### 4. Package Import Errors
**Error**: `ModuleNotFoundError: No module named 'protflex'`

**Fix**:
- Ensure you've installed ProtFlex in development mode: `pip install -e .`
- Check that you're in the correct virtual environment
- Verify the package structure is intact

### 5. PyTorch Version Compatibility
**Error**: `AttributeError: module 'torch' has no attribute...`

**Fix**:
- Update PyTorch to the latest version: `pip install --upgrade torch torchvision`
- Ensure compatibility between PyTorch and CUDA versions

## 🔧 Advanced Usage

### Custom Model Architecture

You can choose between three model architectures:
- `protflex_cnn`: Standard 3D CNN with residual connections
- `dilated_resnet3d`: Deeper network with dilated convolutions
- `multipath_rmsf_net`: Multi-pathway network for capturing different structural aspects

```bash
python -m protflex train --config my_config.yaml --model dilated_resnet3d
```

### Custom Loss Functions

ProtFlex supports several loss functions optimized for RMSF prediction:
- `mse`: Mean Squared Error (default)
- `mae`: Mean Absolute Error
- `rmsf`: Custom RMSF loss combining MSE and RMSE
- `weighted_rmsf`: Weighted loss for emphasizing high-flexibility regions
- `elastic_rmsf`: Elastic net style loss combining L1 and L2 penalties

```bash
python -m protflex train --config my_config.yaml --loss-function weighted_rmsf
```

### Data Augmentation

Enable data augmentation in the configuration file to improve model generalization:

```yaml
training:
  use_augmentation: true
  augmentation_params:
    rotation_prob: 0.5
    rotation_angles: [90, 180, 270]
    flip_prob: 0.3
    noise_prob: 0.2
    noise_scale: 0.05
```

## 📝 Citation

If you use ProtFlex in your research, please cite:

```
@article{protflex2023,
  title={ProtFlex: Deep Learning for Protein Flexibility Prediction},
  author={Felix, S. and et al.},
  journal={Journal of Computational Biology},
  year={2023}
}
```

## 📊 Example Results

After running ProtFlex, you'll get visualizations like these:

- RMSF Profiles: Comparing predicted vs. actual flexibility
- Error Distribution Analysis
- Residue-type Performance Breakdown
- Loss Curves During Training

## 📬 Support

For questions and issues, please open an issue on the GitHub repository or contact the maintainers.

---

