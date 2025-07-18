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

##  Tech stack
- Python 3
- PyTorch
- torchvision.transforms v2
- NumPy
- matplotlib
- tqdm
-  PIL

##  Training Features

-  Loss: Mean Squared Error (MSE)  
-  Metric: Mean Absolute Error (MAE) 
-  `Adam` optimizer  
-  `ReduceLROnPlateau` scheduler  
-  Early stopping: If no improvement for 10 epochs
-  Saving best model `.pt`  
-  Checkpointing: Best model is saved as model_state_dict_XX.pt
