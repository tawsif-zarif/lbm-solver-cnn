# AI Surrogate Thermal Wind Tunnel
# Author: Tawsif Zarif

This repository contains a computational engineering portfolio project that replaces a computationally expensive fluid dynamics simulation with a near-instantaneous Convolutional Neural Network (CNN). 

**📄 Full Documentation:** For a comprehensive explanation of the underlying Lattice Boltzmann equations, thermodynamic boundary conditions, and the neural network methodology, please refer to the full project write-up included in this repository: `AI Surrogate Thermal Wind Tunnel.docx`.

## Core Architecture

### 1. The Physics Engine: Lattice Boltzmann Method (CFD)
Instead of solving the macroscopic Navier-Stokes equations, the physics engine operates on a mesoscopic kinetic model. 
* **Topology:** D2Q9 (2 Dimensions, 9 discrete velocity directions).
* **Collision Operator:** Bhatnagar-Gross-Krook (BGK) relaxation towards localized equilibrium.
* **Boundary Conditions:** A bounce-back (no-slip) condition for fluid momentum on the obstacle surface, combined with a Dirichlet boundary condition fixing the internal obstacle temperature (\(T_{hot} = 2.0\)). 
* **Metric Extraction:** 
  * *Pressure Drag:* Extracted using the LBM's inherent isothermal equation of state (\(p = c_s^2 \rho\)). Pressure drag is isolated by integrating the horizontal components of the boundary pressure field.
  * *Convective Heat Transfer (\(h\)):* Calculated via Fourier’s Law of Conduction at the boundary nodes, leveraging the zero-velocity condition at the wall to equate conductive heat flux to convective dissipation.

### 2. Procedural Geometry Generation
To provide varied training data for the neural network, the pipeline generates randomized 2D bluff bodies.
* Uniformly spaced angles ($0$ to \(2\pi\)) are assigned randomized radii between 5 and 25 units.
* The coordinates are mapped into a bounding polygon using `matplotlib.path`.
* The geometry is converted into a binary boolean mask (1 for obstacle, 0 for fluid) and injected into the 100x400 simulation grid.
* The engine runs the CFD simulation, calculates Drag and \(h\), and appends the metrics to a master CSV while saving the geometry mask as an `.npy` file.

### 3. The PyTorch Surrogate CNN
The surrogate model is a custom PyTorch architecture designed for multi-target regression mapping 2D spatial features to physical constants.
* **Input:** A single-channel `100x400` boolean tensor.
* **Spatial Extraction (Convolutions):** Three sequential `Conv2d` layers (16, 32, and 64 channels respectively) using \(3 \times 3\) sliding kernels. These layers act as automated edge and curve detectors.
* **Non-Linearity & Compression:** Each convolution is followed by a `ReLU` activation to capture the non-linear nature of fluid dynamics, and a `MaxPool2d` operation to downsample the spatial grid, discarding empty space and isolating dominant aerodynamic features.
* **Regression Brain:** The highly compressed 3D spatial maps are flattened into a 1D tensor (38,400 features) and passed through a 512-node fully connected `Linear` layer.
* **Output:** An unactivated 2-node final layer predicting the continuous physical metrics (Drag and \(h\)).

## Installation & Setup

**Prerequisites:** Python 3.8+


# Clone the repository
git clone [https://github.com/YourUsername/ai-surrogate-thermal-wind-tunnel.git](https://github.com/YourUsername/ai-surrogate-thermal-wind-tunnel.git)
cd ai-surrogate-thermal-wind-tunnel

# Install dependencies
pip install numpy pandas matplotlib torch torchvision
Usage1. Generate the DatasetRun the LBM simulation to generate the procedural dataset. This will populate the dataset_geometries/ directory with .npy masks and generate dataset_metrics.csv.Bashpython generate_dataset.py
(Note: Generating a sufficient dataset—e.g., 500 iterations—is computationally intensive and may take several hours on a standard CPU).2. Train the Surrogate ModelOnce the dataset is populated, execute the training script. This script automatically handles the PyTorch Dataset loading, train/test splitting, and backpropagation loop. python train_cnn.py


# Known Limitations & Future WorkData Normalization
In the current implementation, the target metrics are not normalized before entering the CNN. Because aerodynamic drag variance ($\sim 0.4$) is an order of magnitude larger than the variance of the convective heat transfer coefficient ($\sim 0.03$), the Mean Squared Error loss function disproportionately optimizes for drag. This leads to slight underfitting on the thermal predictions. Future iterations will implement standard score normalization ($z = (x - \mu) / \sigma$) on the training labels to ensure equal gradient weighting across both physical domains.
Dataset Size: The current proof-of-concept relies on roughly 460 procedurally generated geometries. Scaling the dataset to 5,000+ examples would likely eliminate remaining generalization errors.
