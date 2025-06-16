You are givne a neural network with the following architecture:

```
Input: 100 x 100 x 3 Image

Layers:
1. Conv(in_channels=3, out_channels=5, kernel_size=3, stride=1, padding=0)
2. MaxPool2d(kernel_size=2, stride=2, padding=0)
3. Conv(in_channels=5, out_channels=10, kernel_size=3, stride=1, padding=0)
4. MaxPool2d(kernel_size=2, stride=2, padding=0)
5. Conv(in_channels=10, out_channels=5, kernel_size=3, stride=1, padding=0)
6. Flatten()
7. MLP(neurons=20)
8. MLP(neurons=10)
```

(a) What is the dimensionality of the activations after each layer.

(b) How many parameters does this network have?