#  Coordinate Regression with CNN in PyTorch

A PyTorch-based Convolutional Neural Network (CNN) that predicts **(x, y)** coordinates from images. Includes custom dataset loader, training loop, validation tracking, early stopping, and model checkpointing.

---

Images should be in the dataset folder.
- `coords.json` maps each image filename to its target coordinates.

---

##  Model Architecture

```text
Input (1, 64, 64)  ⟶ Conv2d(1→32) ⟶ ReLU ⟶ MaxPool
                  ⟶ Conv2d(32→64) ⟶ ReLU ⟶ MaxPool
                  ⟶ Flatten
                  ⟶ Linear(16384→128) ⟶ ReLU
                  ⟶ Linear(128→2) → (x, y)


---

### Tech Stack

- Python 3
- PyTorch
- torchvision.transforms v2
- NumPy
- matplotlib
- tqdm
- PIL

---

### Training Features

- **Loss**: Mean Squared Error (MSE)  
- **Metric**: Mean Absolute Error (MAE)  
- **Optimizer**: Adam  
- **Scheduler**: ReduceLROnPlateau  
- **Early Stopping**: After 10 epochs w/o improvement  
- **Checkpointing**: Saves best model as `.pt`
