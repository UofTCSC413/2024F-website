# CSC413 Lab 7: Transfer Learning and Descent

Transfer learning is a technique where we use neural network weights trained
to complete one task to complet a different task.
In this tutorial, we will go through an example of *transfer learning* to 
detect American Sign Language (ASL) gestures letters A-I.
Although we could train a CNN from scratch,
you will see that using CNN weights that are pretrained on a larger dataset and
more complex task provides much better results, all with less training.

American Sign Language (ASL) is a complete, complex language that employs signs made by 
moving the hands combined with facial expressions and postures of the body. 
It is the primary language of many North Americans who are deaf and is one of several 
communication options used by people who are deaf or hard-of-hearing.

The hand gestures representing English alphabets are shown below. This lab focuses on 
classifying a subset of these hand gesture images using convolutional neural networks.
Specifically, given an image of a hand showing one of the letters A-I, we want to detect
which letter is being represented.

<img src="https://qualityansweringservice.com/wp-content/uploads/2010/01/images_abc1280x960.png" width=400px" />

By the end of this lab, you will be able to:

1. Analyze the role of batch normalization and other model architecture choice in a neural network.
2. Define the double descent phenomenon and explain why it occurs.
3. Analyze the shape of the training curve of a convolutional neural network with respect to the double descent phenomenon.
4. Apply transfer learning to solve an image classification task.
5. Compare transfer learning vs. training a CNN from scratch.
6. Identify and suggest corrections for model building issues by inspecting misclassified data.


Acknowledgements:

- Data is collected from a previous machine learning course APS360. Only data
  of students who provided consent is included.

Please work in groups of 1-2 during the lab.

## Submission

If you are working with a partner, start by creating a group on Markus.
If you are working alone,
click "Working Alone".

Submit the ipynb file `lab07.ipynb` on Markus 
**containing all your solutions to the Graded Task**s.
Your notebook file must contain your code **and outputs** where applicable,
including printed lines and images.
Your TA will not run your code for the purpose of grading.

For this lab, you should submit the following:

- Part 1. Your answer to the question about the splitting of the data into train/validation/test sets. (1 point)
- Part 2. Your comparison of the CNN model with and without batch normalization. (1 point)
- Part 2. Your comparison of `BatchNorm1d` vs `BatchNorm2d`.  (1 point)
- Part 2. Your analysis of the effect of varying the CNN model width. (1 point)
- Part 2. Your analysis of the effect of varying weight decay parameter. (1 point)
- Part 2. Your analysis of the training curve that illustrates double descent. (1 point)
- Part 3. Your implementation of `LinearModel` for transfer learning. (1 point)
- Part 3. Your comparison of transfer learning vs the CNN model. (1 point)
- Part 4. Your analysis of the confusion matrix. (1 point)
- Part 4. Your explanation for how to mitigate an issue we notice by visually inspecting misclassified images. (1 point)


```
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models, torchvision.datasets

%matplotlib inline
```

## Part 1. Data

We will begin by downloading the data onto Google Colab.

```
# Download lab data file
!wget https://www.cs.toronto.edu/~lczhang/413/asl_data.zip
!unzip asl_data.zip
```

The file structure we use is intentional,
so that we can use `torchvision.datasets.ImageFolder`
to help load our data and create labels.

You can read what `torchvision.datasets.ImageFolder` does for us here
https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder


```python
train_path = "asl_data/train/" # edit me
valid_path = "asl_data/valid/" # edit me
test_path = "asl_data/test/"   # edit me

train_data = torchvision.datasets.ImageFolder(train_path, transform=torchvision.transforms.ToTensor())
valid_data = torchvision.datasets.ImageFolder(valid_path, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.ImageFolder(test_path, transform=torchvision.transforms.ToTensor())
```

As in previous labs,
we can iterate through the one training data point at a time like this:

```python
for x, t in train_data:
    print(x, t)
    plt.imshow(x.transpose(2, 0).transpose(0, 1).numpy()) # display an image
    break # uncomment if you'd like
```

**Task**: What do the variables `x` and `t` contain? What is the shape of our images?
What are our labels? Based on what you learned in Part (a), how were the
labels generated from the folder structure?

```
# Your explanation goes here
```

We saw in the earlier tutorials that PyTorch has a utility to help us
creat minibatches with our data. We can use the same DataLoader helper
here:

```python
train_loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True)

for x, t in train_loader:
    print(x, t)
    break # uncomment if you'd like
```
**Task**: What do the variables `x` and `t` contain? What are their shapes?
What data do they contain?

```
# Your explanation goes here
```

**Task**: How many images are there in the training, validation, and test sets?

```
# Your explanation goes here
```

Notice that there are *fewer* images in the training set, compared to the validation and test sets.
This is so that we can explore the effect of having a limited training set.

**Graded Task**: The data set is generated by students taking pictures of their hand
while making the corresponding gestures. We therefor split the 
training, validation, and test sets were split so that images generated by
a student all belongs in a single data set. In other words, we avoid cases where
some students' images are in the training set and others end up in the test set. 
Why do you think this important for obtaining a representative test accuracy?

```
# Your explanation goes here
```

## Part 2. Training a CNN Model

For this part, we will be working with this CNN network.

```python
class CNN(nn.Module):
    def __init__(self, width=4, bn=True):
        """
        A 4-layer convolutional neural network. The first layer has
        `width` number of channels, and with each layer we half the
        feature width/height and double the number of channels.

        If `bn` is set to False, then batch normalization will not run.
        """
        super(CNN, self).__init__()
        self.width = width
        self.bn = bn
        # define all the conv layers
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=self.width,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.width,
                               out_channels=self.width*2,
                               kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv2d(in_channels=self.width*2,
                               out_channels=self.width*4,
                               kernel_size=3,
                               padding=1)
        self.conv4 = nn.Conv2d(in_channels=self.width*4,
                               out_channels=self.width*8,
                               kernel_size=3,
                               padding=1)
        # define all the BN layers
        if bn:
            self.bn1 = nn.BatchNorm2d(self.width)
            self.bn2 = nn.BatchNorm2d(self.width*2)
            self.bn3 = nn.BatchNorm2d(self.width*4)
            self.bn4 = nn.BatchNorm2d(self.width*8)
        # pooling layer has no parameter, so one such layer
        # can be shared across all conv layers
        self.pool = nn.MaxPool2d(2, 2)
        # FC layers
        self.fc1 = nn.Linear(self.width * 8 * 14 * 14, 100)
        self.fc2 = nn.Linear(100, 9)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        if self.bn:
            x = self.bn1(x)
        x = self.pool(torch.relu(self.conv2(x)))
        if self.bn:
            x = self.bn2(x)
        x = self.pool(torch.relu(self.conv3(x)))
        if self.bn:
            x = self.bn3(x)
        x = self.pool(torch.relu(self.conv4(x)))
        if self.bn:
            x = self.bn4(x)
        x = x.view(-1, self.width * 8 * 14 * 14)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

**Task**:

The training code is written for you. Train the `CNN()` model for at least 6 epochs, and report
on the maximum validation accuracy that you can attain.

As your model is training, you might want to move on to the next question.

```python
def get_accuracy(model, data, device="cpu"):
    loader = torch.utils.data.DataLoader(data, batch_size=256)
    model.to(device)
    model.eval() # annotate model for evaluation (important for batch normalization)
    correct = 0
    total = 0
    for imgs, labels in loader:
        labels = labels.to(device)
        output = model(imgs.to(device))
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]
    return correct / total

def train_model(model,
                train_data,
                valid_data,
                batch_size=64,
                weight_decay=0.0,
                learning_rate=0.001,
                num_epochs=50,
                plot_every=20,
                plot=True,
                device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    model = model.to(device) # move model to GPU if applicable
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay)
    # for plotting
    iters, train_loss, train_acc, val_acc = [], [], [], []
    iter_count = 0 # count the number of iterations that has passed

    try:
        for epoch in range(num_epochs):
            for imgs, labels in iter(train_loader):
                if imgs.size()[0] < batch_size:
                    continue
                labels = labels.to(device)
                imgs = imgs.to(device)
                model.train()
                out = model(imgs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                iter_count += 1
                if iter_count % plot_every == 0:
                    loss = float(loss)
                    tacc = get_accuracy(model, train_data, device)
                    vacc = get_accuracy(model, valid_data, device)
                    print("Iter %d; Loss %f; Train Acc %.3f; Val Acc %.3f" % (iter_count, loss, tacc, vacc))

                    iters.append(iter_count)
                    train_loss.append(loss)
                    train_acc.append(tacc)
                    val_acc.append(vacc)
    finally:
        plt.figure()
        plt.plot(iters[:len(train_loss)], train_loss)
        plt.title("Loss over iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")

        plt.figure()
        plt.plot(iters[:len(train_acc)], train_acc)
        plt.plot(iters[:len(val_acc)], val_acc)
        plt.title("Accuracy over iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.legend(["Train", "Validation"])
```

**Task**: Run the training code below. What validation accuracy can be achieved by this CNN?

```python
cnn = CNN(width=4)
train_model(cnn, train_data, valid_data, batch_size=64, learning_rate=0.001, num_epochs=50, plot_every=25)
```

## Part 2. Model Architecture, Bias/Variance and Double Descent

In this section, we will explore the effect of various aspects of
a CNN model architecture. We will pay particluar attention to 
architecture decisions that affect the bias and variance of the
model. Finally, we explore a phenomenon called **double descent**.


To begin, let's explore the effect of batch normalization.

**Task**: Run the training code below to explore the effect of training *without* batch normalization.

```python
cnn = CNN(bn=False)
train_model(cnn, train_data, valid_data, batch_size=64, learning_rate=0.001, num_epochs=50, plot_every=25)
```

**Graded Task**: Compare the two sets of training curves above for the CNN model with and without
batch normalization. What is the effect of batch normalization on the training loss and accuracy?
What about the validation accuracy?

```
# TODO: Include your analysis here
# SOLUTION - training loss/acc improves a lot more quickly with BN
# SOLUTION   but final val loss is about the same for this task
```

**Graded Task**:
We used the layer called `BatchNorm2d` in our CNN.
What do you think is the difference between `BatchNorm2d` and `BatchNorm1d`?
Why are we using `BatchNorm2d` in our CNN? Why would we use `BatchNorm1d` in an MLP?
You may wish to consult the PyTorch documentation. (How can you find it?)

```
# Explain your answer here
```

**Task**: Run the training code below to explore the effect of varying the model width
for this particular data set.

```python
cnn = CNN(width=2, bn=False)
train_model(cnn, train_data, valid_data, batch_size=64, learning_rate=0.001, num_epochs=50, plot_every=25)
```

```python
cnn = CNN(width=4, bn=False)
train_model(cnn, train_data, valid_data, batch_size=64, learning_rate=0.001, num_epochs=50, plot_every=25)
```

```python
cnn = CNN(width=16, bn=False)
train_model(cnn, train_data, valid_data, batch_size=64, learning_rate=0.001, num_epochs=50, plot_every=25)
```


**Graded Task**: What is the effect of varying the model width above for this particular data set?
Do you notice an effect on the training loss? What about the training/validation accuracy?
The final validation accuracy?
(Your answer may or may not match your expectations. Please answer based on the actual results above.)

```python
# TODO: Include your analysis here
```

**Task**: Run the training code below to explore the effect of weight decay when training a large model.

```
cnn = CNN(width=16, bn=False)
train_model(cnn, train_data, valid_data, batch_size=64, learning_rate=0.001, num_epochs=50, plot_every=25, weight_decay=0.001)
```

```
cnn = CNN(width=16, bn=True) # try with batch norm on
train_model(cnn, train_data, valid_data, batch_size=64, learning_rate=0.001, num_epochs=50, plot_every=25, weight_decay=0.001)
```

```
cnn = CNN(width=16, bn=True) # try decreasing weight decay parameter
train_model(cnn, train_data, valid_data, batch_size=64, learning_rate=0.001, num_epochs=50, plot_every=25, weight_decay=0.0001)
```

**Graded Task**: What is the effect of setting weight decay to the above value?
Do you notice an effect on the training loss? What about the training/validation accuracy?
The final validation accuracy?
(Again, your answer may or may not match your expectations. Please answer based on the actual results above.)

```python
# TODO: Include your analysis here
```

**Task**: Note that there is quite a bit of noise in the results that we might obtain above.
That is, if you run the same code twice, you may obtain different answers.
Why might that be? What are two sources of noise/randomness?

```python
# TODO: Include your explanation here
# SOLUTION: SGD and initalization of parameters
```

These settings that we have been exporting are hyperparameters that should
be tuned when you train a neural network. These hyperparameters interact with
one another, and thus we should tune them using the **grid search** strategy
mentioned in previous labs.

You are not required to perform grid search for this lab, so that we can
explore a few other phenomena.

One interesting phenomenon is called **double descent**. In statistical learning theory,
we expect validation error to *decrease* with increase model capacity, and then *increase*
as the model overfits to the number of data points available for training.
In practise, in neural networks, we often see that as model capacity increases,
validation error first decreases, then increase, and then **decrease again**---hence
the name "double descent".

In fact, the increase in validation error is actually quite subtle.
However, what is readily apparent is that in most cases, increasing
model capacity does *not* result in a decrease in validation accuracy.

**Optional Task**: To illustrate that validation accuracy is unlikely to decrease
with increased model parameter, train the below network. 

```
# Uncomment to run. 
# cnn = CNN(width=40, bn=True)
# train_model(cnn, train_data, valid_data, batch_size=64, learning_rate=0.001, num_epochs=50, plot_every=50)
```

Double descent is actually not that mysterious. It comes from the fact that 
when capacity is large enough there are many parameter choices that achieves 100% training accuracy,
the neural network optimization procedure is effectively choosing a *best parameters*
out of the many that can achieve this perfect training accuracy. This differs from
when capacity is low, where the optimization process needs to find a set of parameter choices that
best fits the training data---since no choice of parameters fits the training data perfectly.
When the capacity is just large enough to be able to find parameters that fit the data,
but too small for there be a range of parameter choices available to be able to select a "best" one.

This twitter thread written by biostatistics professor Daniela Witten
also provides an intuitive explanation, using polynomial curve fitting
as an example: [https://twitter.com/daniela_witten/status/1292293102103748609](https://twitter.com/daniela_witten/status/1292293102103748609)

Double descent explored in depth in this paper here:
[https://openreview.net/pdf?id=B1g5sA4twr](https://openreview.net/pdf?id=B1g5sA4twr)
This paper highlights that the increase in validation/test error occurs 
when the training accuracy approximates 100%.
Moreover, the double descent phenomena is noticable when varying model capacity (e.g. number of parameters)
and when varying the number of iterations/epochs of training.

We will attempt to explore the latter effect---i.e. we will train a large model, use a small
numer of training data points, and explore how each iteration of training impacts validation accuracy.
The effect is subtle and, depending on your neural network initialization, you may not see an effect.
So, a training curve is also provided for you to analyze.

**Optional Task**: Run the code below to try and reproduce the "double descent" phenomena.
This code will take a while to run, so you may wish to continue with the remaining questions
while it runs.

```
# use a subset of the training data
# uncomment to train

# train_data_subset, _ =  random_split(train_data, [50, len(train_data)-50])
# cnn = CNN(width=20)
# train_model(cnn,
#             train_data_subset,
#             valid_data,
#             batch_size=50, # set batch_size=len(train_data_subset) to minimize training noise
#             num_epochs=200,
#             plot_every=1,  # plot every epoch (this is slow)
#             learning_rate=0.0001)  # choose a low learning rate
```

For reference, here is the our training curve showing the loss and accuracy over 200 iterations:

<img src="https://www.cs.toronto.edu/~lczhang/413/double_descent_loss.png" width=400>
<img src="https://www.cs.toronto.edu/~lczhang/413/double_descent.png" width=400>

We are not able to consistently reproduce this result (e.g., due to initialization),
so it is totally reasonable for your figure to look different!


**Task**: In the provided training curve,
during which iterations do the validation accuracy initially increase
(i.e. validation error decrease)?

```
# TODO: Include your answer here
```

**Graded Task**: In the provided training curve,
during which iterations do the validation accuracy decrease slightly?
Approximately what training accuracy is achieved at this piont?

```
# TODO: Include your answer here
```

**Task**: In the provided training curve,
during which iterations do the validation accuracy increase for a second time
(i.e. validation error descends for a second time)?

```
# TODO: Include your answer here
```

## Part 3. Transfer Learning

For many image classification tasks, it is generally not a good idea to train a
very large deep neural network model from scratch due to the enormous compute
requirements and lack of sufficient amounts of training data.

A better option is to try using an existing model that performs a
similar task to the one you need to solve. This method of utilizing a
pre-trained network for other similar tasks is broadly termed
**Transfer Learning**. In this assignment, we will use Transfer Learning
to extract features from the hand gesture images. Then, train a smaller
network to use these features as input and classify the hand gestures.

As you have learned from the CNN lecture, convolution layers extract various
features from the images which get utilized by the fully connected layers
for correct classification. AlexNet architecture played a pivotal role in
establishing Deep Neural Nets as a go-to tool for image classification
problems and we will use an ImageNet pre-trained AlexNet model to
extract features in this assignment.

Here is the code to load the AlexNet network, with pretrained weights.
When you first run the code, PyTorch will download the pretrained weights
from the internet.


```python
import torchvision.models
alexnet = torchvision.models.alexnet(pretrained=True)

print(alexnet)
```

As you can see, the `alexnet` model is split up into two components:
`alexnet.features` and 
`alexnet.classifier`.  The first neural network component, `alexnet.features`,
is used to
computed convolutional features, which is taken as input in `alexnet.classifier`.

The neural network `alexnet.features` expects an image tensor of shape
Nx3x224x224 as inputs and it will output a tensor of shape Nx256x6x6 . (N = batch size).

Here is an example code snippet showing how you can compute the AlexNet
features for some images (your actual code might be different):

```python
img, label = train_data[0]
features = alexnet.features(img.unsqueeze(0)).detach()

print(features.shape)
```

Note that the `.detach()` at the end will be necessary in your code. The reason is that
PyTorch automatically builds computation graphs to be able to backpropagate
graidents. If we did not explicitly "detach" this tensor from the AlexNet portion
of the computation graph, PyTorch might try to backpropagate gradients to the AlexNet
weight and tune the AlexNet weights.

**TODO** Compute the AlexNet features for each of your training, validation, and test data
by completing the function `compute_features`.
The code below creates three new arrays called `train_data_fets`, `valid_data_fets`
and `test_data_fets`. Each of these arrays contains tuples of the form 
`(alexnet_features, label)`.

```python
def compute_features(data):
    fets = []
    for img, t in data:
        features = None  # TODO
        features = alexnet.features(img.unsqueeze(0)).detach() # SOLUTION
        fets.append((features, t),)
    return fets

train_data_fets = compute_features(train_data)
valid_data_fets = compute_features(valid_data)
test_data_fets = compute_features(test_data)
```

In the rest of this part of the lab, we will test two models that
will take **as input** these AlexNet features, and make a prediction
for which letter the hand gesture represents.
The two models are
a linear model, a two-layer MLP.
We will compare the performance of these two models.

**Graded Task**: Complete the definition of the `LinearModel` class,
which is a linear model (e.g., logistic regression).
This model should as input these AlexNet features, and make a prediction
for which letter the hand gesture represents.

```python
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # TODO: What layer need to be initialized?
        self.fc = nn.Linear(256 * 6 * 6, 10) # SOLUTION

    def forward(self, x):
        x = x.view(-1, 256 * 6 * 6) # flatten the input
        z = None # TODO: What computation needs to be performed?
        z = self.fc(x) # SOLUTION
        return z

m_linear = LinearModel()
m_linear(train_data_fets[0][0]) # this should produce a(n unnormalized) prediction
```

**Task**:
Train a `LinearModel()` for at least 6 epochs, and report
on the maximum validation accuracy that you can attain.
We should still be able to use the `train_model` function, but
make sure to provide the AlexNet features as input (and not the
image features).


```python
m_linear = LinearModel()
# TODO: Train the linear model. Include your output in your submission
train_model(m_linear, train_data_fets, valid_data_fets) # SOLUTION
```

**Graded Task**: Compare this model with the CNN() models that we trained
earlier. How does this model perform in terms of validation accuracy?
What about in terms of the time it took to train this model?

```
# TODO: Your observation goes here
```

**Task**:
We decide to use AlexNet features as input to our MLP, and avoided tuning AlexNet
weights. However, we could have considered AlexNet to be a part of our model, and
continue to tune AlexNet weights to improve our model performance. What are the
advantages and disadvantages of continuing to tune AlexNet weights?

```
# TODO
```

## Part 4. Data

**Task**: Report the test accuracy on this transfer learning model.

```
# TODO
get_accuracy(m_linear, test_data_fets) # SOLUTION
```

**Task**: Use this code below to construct the confusion matrix for this model
over the test set.

```
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import sklearn
label = "ABCDEFGHI"
def plot_confusion(model, data):
    n = 0
    ts = []
    ys = []
    for x, t in data:
        z = model(x.unsqueeze(0))
        y = int(torch.argmax(z))
        ts.append(t)
        ys.append(y)

    cm = confusion_matrix(ts, ys)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label)
    plt.figure()
    disp.plot()

plot_confusion(m_linear, test_data_fets)
```
**Graded Task**: Which class is most likely mistaken as another?
Is this reasonable? (i.e. is that class particularly challenging, or 
very similar to another class?)

```
# TODO: Include your analysis here
```

**Task**: In order to understand where errors come from, it is *crucial* that
we explore why and how our models fail. A first step is to visually inspect the
test data points where failure occurs. That way, we can identify what we can do 
to prevent/fix errors before our models are deployed.

Run the below code to display images in the test set that our model *misclassifies*:

```
for i, (x, t) in enumerate(test_data_fets):
    y = int(torch.argmax(m_linear(x)))
    if not (y == t):
        plt.figure()
        plt.imshow(test_data[i][0].transpose(0,1).transpose(1,2).numpy())
```

**Task**: By visually inspecting these misclassified images, we see that there are
two main reasons for misclassification. What reason for misclassification is
due to a mistake in the formatting of the test set images?

```
# TODO
```

**Graded Task**: We also see a much more serious issue, where gestures made by
individuals with darker skin tones may be more frequently misclasified.
This result suggests that errors in the model may impact some groups more than
others. What steps should we take to mitigate this issue?

```
# TODO
```


