First, we hvae to load the `alexnet` model which is part of torchvision:

```python
import torchvision
alexnet = torchvision.models.alexnet()
```

The number of parameters for the entire model, is the easier part: We can simply use the `parameters()` iterator which returns the set of parameters for each module. Those can then be counted using the `numel()` method resulting in

```python
sum(p.numel() for p in alexnet.parameters())
```
Obtaining the number of parameters for each of the layer requires looking into the source code of the alexnet model. The structure is similar to vgg and the forward pass looks like this:

```python
x = self.features(x)
x = self.avgpool(x)
x = torch.flatten(x, 1)
x = self.classifier(x)
```

A closer look at the implementation reveals that we can obtain the individual layers parameter by simply iterating over the `self.features` and `self.classifier` modules. The `self.avgpool` module does not have any parameters. The following code snippet shows how to obtain the number of parameters for each layer of the convolutional backbone:

```python
for layer in alexnet.features:
    print(layer, sum(p.numel() for p in layer.parameters()))
```

To get the number of paramers in the cnn head, simply update the code snippet to iterate over `alexnet.classifier` instead of `alexnet.features`.