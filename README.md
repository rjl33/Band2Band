# Band2Band: Multi-Label Bandgap Prediction in Metamaterials Using CNNs

A deep learning framework using ResNet18-based transfer learning for rapid multi-label prediction of bandgap existence across five frequency ranges in 2D acoustic metamaterials. This project achieves **94.5% element-wise accuracy**, exceeding the established benchmark of 91.48% and enabling ~25 ms inference per sample.

## Overview

Acoustic metamaterials exhibit unique wave propagation properties characterized by frequency bandgaps—ranges where mechanical waves cannot propagate. Traditional finite element simulations require hours per design, severely limiting design space exploration. This project demonstrates that convolutional neural networks with transfer learning can rapidly and accurately predict bandgap presence across multiple frequency ranges from metamaterial unit cell geometries.

**Authors**: Ryan Lengacher, Jiaxuan Zhang, Ukamaka Ezimora (Duke University, December 2024)
  

## Project Report

📄 **Full Technical Report**: [PDF](Metamaterial_BandGap_Preditions_using_CNNs.pdf)

The report includes:
- Comprehensive methodology and two-phase training strategy
- Systematic ablation studies evaluating all architectural modifications
- Detailed per-class performance analysis across five frequency ranges
- Grad-CAM visualizations showing learned geometric feature attention
- Discussion of limitations and future research directions

### Key Achievements

- **94.5% element-wise accuracy** across five frequency ranges on test data
- **3.0 percentage point improvement** over Chen et al.'s interpretable baseline (91.48%)
- **Multi-label classification**: Simultaneous prediction across [0-1], [1-2], [2-3], [3-4], [4-5] kHz ranges
- **Rapid inference**: 25.54 ms per sample (~39 samples/second), enabling design space exploration
- **Systematic ablation studies**: Evaluated ResNet18 vs ResNet50, resolution scaling, loss functions, and regularization
- **Transfer learning optimization**: Two-phase training strategy with successful feature adaptation from ImageNet

## Problem Statement

Given a 2D representation of an acoustic metamaterial unit cell, predict whether complete bandgaps exist across five specific frequency ranges: [0-1], [1-2], [2-3], [3-4], and [4-5] kHz. This is formulated as a multi-label binary classification problem, where each sample may exhibit bandgaps in multiple frequency ranges simultaneously.

**Challenge**: Traditional finite element eigenvalue analysis requires hours per geometry and extensive computational resources, limiting design space exploration.

**Our Solution**: A ResNet18-based CNN that provides near-instantaneous predictions (<26 ms per geometry), enabling rapid screening of thousands of metamaterial candidates while exceeding the accuracy of interpretable baseline methods.

## Methodology

### Dataset

This project uses the acoustic metamaterial dataset from:

**Chen, Z., Ogren, A., Daraio, C., Brinson, L.C., Rudin, C.** (2022). "How to see hidden patterns in metamaterials with interpretable machine learning." *Materials Today Communications*, 33, 104565. https://doi.org/10.1016/j.mtcomm.2022.104565

Their interpretable LightGBM model achieved **91.48% accuracy**, which we surpass through transfer learning and systematic optimization to achieve **94.5% accuracy**.

**Dataset characteristics:**
- **Size**: 32,768 samples with dispersion data (20 bands × 150 k-points per sample)
- **Input**: 10×10 unit cell geometries (binary: soft polymer vs. stiff steel)
- **Symmetry**: Four-axis symmetry reduces 100 pixels to 15 irreducible pixels (2^15 design space)
- **Materials**: Soft polymer (E=2 GPa, ρ=1000 kg/m³) and stiff steel (E=200 GPa, ρ=8000 kg/m³)
- **Output**: Multi-label binary classification for five 1 kHz frequency ranges
- **Preprocessing**: 10×10 patterns reconstructed via symmetry, upsampled to 128×128, normalized to [-1, 1]
- **Split**: 70% training (22,938), 15% validation (4,915), 15% test (4,915)

### Systematic Ablation Studies

The project employed a rigorous systematic evaluation approach, testing modifications individually before combining them:

**Baseline Model** (87.4% test accuracy):
- ResNet18 with averaged RGB→grayscale first conv layer
- 64×64 input resolution
- Two-phase training: 5-epoch warmup (frozen backbone) + fine-tuning
- Class-balanced binary cross-entropy loss

**Individual Modifications Tested**:
1. **Class-weighted loss**: Aggressive weighting [1.0, 1.0, 1.2, 1.8, 2.5] → **Decreased to 86.4%** (over-predicted high-frequency bandgaps)
2. **Higher resolution (128×128)**: → **89.7% test accuracy** (+2.3 pp improvement)
3. **Deeper architecture (ResNet50)**: → **89.6% test accuracy** (+2.2 pp improvement)
4. **Combined ResNet50 + 128×128**: → **86.4%** (model too deep for dataset size)
5. **Added dropout (0.35)**: → **86.2%** (when combined with ResNet50)

**Final Optimized Model** (94.5% test accuracy):
- ResNet18 architecture (optimal for dataset size)
- 128×128 input resolution (captures fine geometric details)
- Per-class threshold optimization (0.40, 0.50, 0.50, 0.60, 0.70 for five classes)
- Dropout 0.35 after final FC layer
- Extended training to 30 epochs
- Two-phase strategy: warmup + fine-tuning with ReduceLROnPlateau

### Training Strategy

- **Data Augmentation**: Rotation, scaling, and translation to enhance generalization
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout and weight decay to prevent overfitting
- **Cross-validation**: K-fold validation for robust performance estimation
- **Early stopping**: Monitoring validation loss to prevent overfitting

### Ablation Studies

Systematic component testing revealed:
- Transfer learning provides strong baseline performance (87%)
- Custom architectural modifications enable >90% accuracy
- Fine-resolution inputs improve boundary feature detection
- Ensemble methods further boost performance to 94.5%

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)

### Dependencies

```bash
pip install torch torchvision
pip install numpy matplotlib scikit-learn
pip install pandas pillow
```

## Usage

### Data Loading

The `Data_Loader.py` module handles dataset preparation and augmentation:

```python
from Data_Loader import MetamaterialDataset, get_dataloaders

# Load data
train_loader, val_loader, test_loader = get_dataloaders(
    data_path='./data',
    batch_size=32,
    train_split=0.7,
    val_split=0.15
)
```

### Training a Model

#### Simple CNN

```python
python Full_Simple_CNN.py --epochs 50 --batch_size 32 --lr 0.001
```

#### Transfer Learning (ResNet18)

```python
python Trasnfer_Learning_CNN.py --epochs 50 --freeze_backbone --fine_tune_epoch 30
```

#### Final Optimized Model

```python
python Final_Model.py --epochs 100 --batch_size 64 --lr 0.0005
```

### Inference

```python
import torch
from Final_Model import BandgapCNN

# Load trained model
model = BandgapCNN()
model.load_state_dict(torch.load('bandgap_cnn_simple.pth'))
model.eval()

# Predict bandgap
with torch.no_grad():
    prediction = model(metamaterial_image)
    has_bandgap = prediction > 0.5
```

### Model Configuration

The final model configuration is stored in `Final_Model.json` for reproducibility:

```python
import json

with open('Final_Model.json', 'r') as f:
    config = json.load(f)
```

## File Structure

```
Band2Band/
├── Data_Loader.py                          # Dataset and dataloader utilities
├── Full_Simple_CNN.py                      # Baseline CNN architecture
├── Trasnfer_Learning_CNN.py               # ResNet18 transfer learning
├── Transfer_Learning_CNN_Resnet50.py      # ResNet50 transfer learning
├── Transfer_Learning_Finer_Resolution.py  # High-resolution input model
├── Transfer_Learning_class_loss.py        # Loss function experiments
├── Final_Model.py                         # Optimized final architecture
├── Final_Model.json                       # Model configuration
├── bandgap_cnn_simple.pth                 # Trained model weights
├── band2band_first_CN.ipynb              # Exploratory notebook
├── data_load.ipynb                        # Data analysis notebook
└── README.md
```

## Results

### Performance Metrics

**Overall Accuracy**:
| Model | Train Acc. | Val. Acc. | Test Acc. |
|-------|-----------|-----------|-----------|
| Baseline (ResNet18, 64×64) | 89.6% | 87.3% | 87.4% |
| Higher Resolution (128×128) | 91.3% | 89.7% | 89.7% |
| ResNet50 | 91.2% | 89.4% | 89.6% |
| **Final Model (30 epochs)** | **99.1%** | **94.5%** | **94.5%** |
| Chen et al. Benchmark | - | - | 91.48% |

**Per-Class F1 Scores** (Final Model):
| Frequency Range (kHz) | [0-1] | [1-2] | [2-3] | [3-4] | [4-5] |
|----------------------|-------|-------|-------|-------|-------|
| **Final Model** | 0.952 | 0.956 | 0.919 | 0.853 | 0.686 |
| Baseline | 0.897 | 0.910 | 0.811 | 0.698 | 0.358 |
| **Improvement** | +0.055 | +0.046 | +0.108 | +0.155 | **+0.328** |
| Optimized Threshold | 0.50 | 0.40 | 0.50 | 0.60 | 0.70 |

**Note**: Higher frequency ranges (3-4 kHz, 4-5 kHz) are more challenging due to subtle geometric features. The dramatic improvement in the [4-5] kHz range (+0.328 F1 score) demonstrates the effectiveness of increased resolution and threshold optimization.

### Key Findings

1. **Resolution is critical**: Increasing input from 64×64 to 128×128 provided the largest single improvement (+2.3 pp), enabling the network to capture fine geometric details at stiff-soft material interfaces

2. **Model depth vs. dataset size**: ResNet50 performed similarly to ResNet18 individually, but combining ResNet50 with higher resolution degraded performance—the model became too deep for 22,938 training samples

3. **Learned feature selectivity**: Grad-CAM visualizations show the 30-epoch model focuses sharply on stiff-soft material interfaces and specific geometric configurations, while earlier checkpoints show diffuse attention across multiple features

4. **Class-specific thresholds matter**: Per-class threshold optimization (0.40, 0.50, 0.50, 0.60, 0.70) addresses tendency toward false positives in higher frequency ranges, providing modest but consistent improvements

5. **High-frequency bandgaps are harder**: The [4-5] kHz range showed dramatic improvement (+0.328 F1) from baseline to final model, but still underperforms lower ranges (0.686 vs. 0.95), likely due to more subtle geometric features

6. **Aggressive class weighting backfires**: Heavy weighting for minority classes caused over-prediction of bandgaps, reducing overall accuracy

### Computational Efficiency

- **Training time**: 10-12 hours on GPU for full 30-epoch optimization
- **Inference time**: **25.54 ms per sample** (~39 samples/second)
- **Speedup vs. FEM**: Enables rapid screening of thousands of candidates vs. hours per finite element simulation
- **Practical impact**: Once trained, the model can evaluate an entire design space exploration in minutes rather than months

## Applications

This framework enables:

### Design Optimization
- Rapid screening of thousands of metamaterial candidates
- Integration with optimization algorithms for automated design
- Multi-objective optimization (bandgap width, frequency, manufacturability)

### Physics Discovery
- Identifying structure-property relationships in metamaterials
- Understanding feature importance through visualization
- Inverse design: generating structures for target bandgaps

### Engineering Applications
- Vibration isolation systems
- Acoustic filters and waveguides
- Elastic wave control in mechanical systems
- Seismic protection structures

## Technical Details

### Model Architecture (Final)

The final model combines:
- **Convolutional blocks**: 5 layers with progressive channel expansion (32→64→128→256→512)
- **Activation**: ReLU with batch normalization after each conv layer
- **Pooling**: Max pooling (2×2) for spatial downsampling
- **Global pooling**: Adaptive average pooling before classification head
- **Classifier**: Two fully connected layers with dropout (p=0.5)
- **Output**: Sigmoid activation for binary classification

### Loss Function

Binary cross-entropy loss with class weighting to handle potential dataset imbalance:

```python
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weight)
```

### Data Preprocessing

- Normalization to zero mean, unit variance
- Geometric augmentations (rotation ±15°, scaling 0.9-1.1×)
- Image size: [original size] → [input size]

## Future Work

**1. Multi-Resolution Generalization**
- Current model limited to 10×10 unit cell discretizations
- Test and retrain on finer grids (20×20, 50×50, 80×80) for higher spatial resolution metamaterial designs
- Enable flexible design at any required level of geometric detail

**2. Inverse Design (Property-to-Structure)**
- Extend beyond bandgap identification to generative design
- Embed model in optimization framework as surrogate for gradient-based search
- Leverage learned geometric selectivity (Grad-CAM insights) to guide design toward viable structures
- Adapt Chen et al.'s high-precision inverse-design approach to CNN-based surrogate

**3. Dataset Augmentation for High-Frequency Ranges**
- Generate additional samples capturing subtle geometries for [3-4] and [4-5] kHz ranges
- Improve F1 scores in underperforming frequency ranges
- Requires significant computational resources for additional FE simulations

**4. Regression Capabilities**
- Predict continuous bandgap width and center frequency (not just binary existence)
- Enable more nuanced material property optimization

**5. 3D Metamaterial Extension**
- Extend framework to volumetric structures beyond 2D unit cells
- Explore 3D convolutional architectures

**6. Physics-Informed Constraints**
- Incorporate wave physics constraints into loss functions
- Improve model reliability and reduce non-physical predictions

## Related Work

This project builds on research in:
- Deep learning for materials science
- Physics-informed machine learning
- Metamaterial design and optimization
- Computer vision for scientific applications

### References

1. **Chen, Z., Ogren, A., Daraio, C., Brinson, L.C., Rudin, C.** (2022). "How to see hidden patterns in metamaterials with interpretable machine learning." *Materials Today Communications*, 33, 104565. https://doi.org/10.1016/j.mtcomm.2022.104565
   - Provides the benchmark dataset and baseline CNN model (91.48% accuracy)
   - Demonstrates interpretable machine learning approaches for metamaterial design

## Reproducibility

All experiments are reproducible using the provided code and configuration files. Random seeds are fixed for deterministic results:

```python
torch.manual_seed(42)
np.random.seed(42)
```

Model checkpoints and training logs are available upon request.

## License

This project is available under the MIT License.

## Contact

Ryan Lutz
ryanjohnlutz@gmail.com
Duke University - Mechanical Engineering and Materials Science  
[GitHub](https://github.com/rjl33)

This work was completed as part of a machine learning course project at Duke University, demonstrating the application of deep learning to computational mechanics and materials design.

---

**Acknowledgments**: Dataset and baseline methodology adapted from Chen et al. (2022). We thank the authors for making their data and approach publicly available. Computational resources provided by Duke Research Computing.
