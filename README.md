# Handwritten Digit Recognition CNN

A Convolutional Neural Network that recognize handwrittern
digits 0-0 with 98.7% accuracy - beating the human baseline 
of 97-98% on the MNIST benchmark dataset.

##Results
- Accuracy: 98.7% on 10.000 test images
- Dataset: 70.000 real handwritten digit images(MNIST)
- Architecture: 2 Conv layers + 2 MaxPool layers + 2 FC layers
- Training time : probably 3 minutes on CPU

## What I learned
- Why CNNs are better than regular neural networks for images
  (a regular network needs billions of weights for one image,
  a CNN slides a small filter — way more efficient)
- How filters automatically learn to detect edges and shapes
  without anyone programming them manually
- What pooling does — keeps the strongest signal, throws away
  exact position, makes the model robust to small shifts
- That 3 epochs of training was enough to beat human performance

## How to run
```bash
git clone https://github.com/doraduc/mnist-digit-cnn
cd mnist-digit-cnn
pip install -r requirements.txt
python mnist_cnn.py
```
The MNIST dataset downloads automatically on first run.

## Architecture
```
Input (28x28 image)
  → Conv2d(1, 16, 3x3) + ReLU + MaxPool
  → Conv2d(16, 32, 3x3) + ReLU + MaxPool
  → Flatten
  → Linear(32x7x7, 128) + ReLU
  → Linear(128, 10)
  → Output (digit 0-9)
```

## Tech stack
Python · PyTorch · torchvision · MNIST

## Key insight
After just 3 training epochs the model hit 98.7% — higher 
than most humans on messy handwriting. This is the same 
technology powering Face ID, Tesla Autopilot cameras, and 
Google Photos face recognition, just at a smaller scale.
