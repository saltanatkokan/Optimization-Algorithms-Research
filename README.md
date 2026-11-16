# Comparative Analysis of Optimization Algorithms

A Jupyter Notebook project that compares five gradient-based optimizers, **SGD, SGD with Momentum, Adam, RMSprop and AdamW**, by training the same fully connected neural network on the **Fashion-MNIST** dataset using **PyTorch**.

**Tech stack:** Python · PyTorch · torchvision · pandas · NumPy · Matplotlib · Jupyter Notebook

---

## Experiment Setup

| Item | Value |
|------|-------|
| Dataset | Fashion-MNIST (60,000 training / 10,000 test images, 10 classes) |
| Preprocessing | `ToTensor()` + `Normalize((0.5,), (0.5,))` |
| Model | Fully connected network (`FashionDNN`): Flatten → Linear(784, 128) → ReLU → ... |
| Loss function | CrossEntropyLoss |
| Batch size | 64 |
| Epochs | 30 |
| Learning rate | 0.01 for SGD and Momentum (momentum = 0.9); 0.001 for Adam, RMSprop and AdamW |

The architecture, data, batch size and number of epochs are identical for every experiment. Only the optimizer changes.

The project has two parts:

1. **Comparative experiment**: each optimizer is trained once, and the final test accuracy and training loss are compared.
2. **Robustness experiment**: each optimizer is trained with three random seeds (`42`, `123`, `2024`) to measure how stable the results are (mean and standard deviation).

## Project Structure

```
Optimization Algorithms Research/
├── optimization_algorithms.ipynb   # Main notebook (all experiments and plots)
├── final_results.csv               # Results of the single-run comparison
├── robustness_results.csv          # Results of the multi-seed robustness experiment
├── housing.csv                     # Additional small dataset (not used in the notebook)
├── models/                         # Folder for saved models (currently empty)
└── data/                           # Fashion-MNIST (downloaded automatically, not in the repo)
```

## Results

### Single run (final results)

| Optimizer | Test accuracy (%) | Final training loss |
|-----------|-------------------|---------------------|
| SGD       | 88.17 | 0.2625 |
| Momentum  | 89.21 | 0.1458 |
| Adam      | 88.39 | 0.1246 |
| RMSprop   | 89.28 | 0.1257 |
| AdamW     | 88.58 | 0.1328 |

### Robustness (3 seeds: 42, 123, 2024)

| Optimizer | Mean accuracy (%) | Std (%) | Mean loss |
|-----------|-------------------|---------|-----------|
| Adam      | 88.66 | 0.22 | 0.125 |
| AdamW     | 88.53 | 0.14 | 0.134 |
| Momentum  | 88.46 | 0.50 | 0.144 |
| RMSprop   | 88.18 | 0.12 | 0.130 |
| SGD       | 87.61 | 0.23 | 0.264 |

### Observations

- In the single run, RMSprop and Momentum scored highest. Averaged over three seeds, **Adam** had the best mean accuracy, so a single run alone can be misleading.
- Plain **SGD** is clearly the weakest: lowest mean accuracy and a training loss about twice as high as the others after 30 epochs.
- The adaptive optimizers (Adam, AdamW, RMSprop) reach a lower training loss than SGD, and RMSprop and AdamW were the most stable across seeds.
- The differences between the top four optimizers are small (around 0.5 percentage points), and with only 3 seeds they should be treated as indicative, not conclusive.

## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/YOUR_USERNAME/Optimization-Algorithms-Research.git
   cd Optimization-Algorithms-Research
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate

   pip install torch torchvision numpy pandas matplotlib jupyter
   ```

3. **Open the notebook**

   ```bash
   jupyter notebook optimization_algorithms.ipynb
   ```

4. **Run all cells.** The Fashion-MNIST dataset is downloaded automatically into the `data/` folder on the first run.

> Training each optimizer for 30 epochs works on CPU, but the robustness experiment (5 optimizers × 3 seeds) can take a while. A GPU makes it faster.

## License

Add a license of your choice (for example MIT) if you plan to share this project publicly.
