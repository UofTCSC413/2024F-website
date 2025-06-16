An example implementation could look like this. First we deinfe the objective function and its gradient:

```python
def objective(x):
    return x**2

def obj_grad(x):
    return 2*x
```

Then, we implement the momentum optimizer itself:

```python
def sgd_with_momentum(obj, grad, x_init, learning_rate, momentum, max_iter):
  x = x_init
  update = 0
  for i in range(max_iter):
    update = momentum * update - learning_rate * grad(x)
    x = x + update

    print('>%d f(%s) = %.5f' % (i, x, obj(x)))
  return x
```

It can now be invoked directly via:
```python
sgd_with_momentum(
    objective, obj_grad, x_init=3.0, learning_rate=0.1, momentum=0.5, 
    max_iter=20
    )
```
