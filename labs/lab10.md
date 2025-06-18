# CSC413 Lab 8: Text Classification using RNNs

**Sentiment Analysis** is the problem of identifying the writer's sentiment given a piece of text.
Sentiment Analysis can be applied to movie reviews, feedback of other forms, emails, tweets, 
course evaluations, and much more.

In this lab, we will build an RNN to classify positive vs negative tweets
We use the Sentiment140 data set, which contains tweets with either a positive
or negative emoticon. Our goal is to determine whether which type of
emoticon the tweet (with the emoticon removed) contained. The dataset was actually collected by
a group of students, much like you, who are doing their first machine learning projects.

By the end of this lab, you will be able to:

- Use PyTorch to train an RNN model
- Apply and analyze the components of an RNN model
- Explain how batching is done on sequence data, where the training data in a batch may have different lengths
- Use pre-trained word embeddings as part of a transfer learning strategy for text classification
- Understand the bias that exists in word embeddings and language models.

Acknowledgements:

- Data is sampled from http://help.sentiment140.com/for-students


Please work in groups of 1-2 during the lab.

## Submission

If you are working with a partner, start by creating a group on Markus. If you are working alone,
click "Working Alone".

Submit the ipynb file `lab10.ipynb` on Markus 
**containing all your solutions to the Graded Task**s.
Your notebook file must contain your code **and outputs** where applicable,
including printed lines and images.
Your TA will not run your code for the purpose of grading.

For this lab, you should submit the following:

- Part 1. Your output showing several positive tweets. (1 point)
- Part 2. Your explanation of the shapes of `wordemb`. (1 point)
- Part 2. Your explanation of the shapes of `h` and `out`. (2 points)
- Part 2. Your explanation of why computing the mean and max of hidden states across *all* time steps is likely more informative than using the final output state. (1 point)
- Part 3. Your demonstration of the model's ability to "overfit" on a data set. (1 point)
- Part 3. Your output from training the model on the full data set. (1 point)
- Part 4. Your explanation of why `MyGloveRNN` requires fewer iteration to obtain "good" accuracy. (1 point)
- Part 4. Your comparison of `MyGloveRNN` and `MyRNN` in low data settings.. (1 point)
- Part 4. Your explanation of where the biases in embeddings come from, and whether our model will have the same sorts of baises.. (1 point)


## Part 1. Data

Start by running these two lines of code to download the data on to Google Colab.

```python
# Download tutorial data files.
!wget https://www.cs.toronto.edu/~lczhang/413/sample_tweets.csv
```

As always, we start by understanding what our data looks like. Notice that the
test set has been set aside for us. Both the training and test set files follow
the same format. Each line in the csv file contains the tweet text,
the string label "4" (positive) or "0" (negative), and some additional information about the tweet.

```python
import csv
datafile = "sample_tweets.csv"

# Training/Validation set
data = csv.reader(open(datafile))
for i, line in enumerate(data):
    print(line)
    if i > 10:
        break
```

**Task**: How many positive and negative tweets are in this file?

```python
# TODO
from collections import Counter # SOLUTION
print(Counter(x[0] for x in csv.reader(open(datafile))))
```

**Graded Task**: We have printed several negative tweets above. 
Print 10 positive tweets.

```python
# TODO: Please make sure to include both your code and the
# printed output
```

We will now split the dataset into training, validation, and test sets:

```python
# read the data; convert labels into integers
data = [(review, int(label=='4'))  # label 1 = positive, 0 = negative
        for label, _, _, _, _, review in csv.reader(open(datafile))]

# shuffle the data, since the file stores all negative tweets first
import random
random.seed(42)
random.shuffle(data)

train_data = data[:50000] 
val_data = data[50000:60000] 
test_data = data[60000:]
```

In order to be able to use neural networks to make predictions about these tweets,
we need to begin by convert these tweets into sequences of numbers, each representing
a words. This is akin to a one-hot encoding: each word will be converted into an
a number representing the unique *index* of that word.

Although we could do this conversion by writing our own python code,
torch has a package called **torchtext** that has utilities useful for text classification
and generation tasks. 
In particular, the `Vocab` class and `build_vocab_from_iterator` will be useful for us
for building the mapping from words to indices.

```python
import torchtext

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab, build_vocab_from_iterator

# we will *tokenize* each word by using a tokenzier from 
# https://pytorch.org/text/stable/data_utils.html#get-tokenizer

tokenizer = get_tokenizer("basic_english")
train_data_words = [tokenizer(x) for x, t in train_data]

# build the vocabulary object. the parameters to this function
# is described below
vocab = build_vocab_from_iterator(train_data_words,
                                  specials=['<bos>', '<eos>', '<unk>', '<pad>'],
                                  min_freq=10)

# set the index of a word not in the vocabulary
vocab.set_default_index(2) # this is the index of the `<unk>` keyword
```

Now, `vocab` is an object of class `Vocab` (see more here [https://pytorch.org/text/stable/vocab.html](https://pytorch.org/text/stable/vocab.html) )
that provides functionalities for converting words into their indices.
In addition to words appearing in the training set, ther are four special tokens that 
we use, akin to placeholder words:

- `<bos>`, to indicate the beginning of a sequence.
- `<eos>`, to indicate the end of a sequence.
- `<unk>`, to indicate a word that is *not* in the vocabulary. This includes
  words that appear too infrequently to be included in the vocabulary, and any
  other words in the validation/test sets that are not see in training.
- `<pad>`, used for padding shorter sequences in a batch: since each tweet
  may have different length, the shorter tweets in each batch will be padded with
  the `<pad>` token so that each sequence (tweet) in a batch has the same length.

The `min_freq` parameter identifies the minimum number of times a word must appear in the
training set in order to be included in the vocabulary.

Here you can see the `vocab` object in action:

```python
# Print the number of words in the vocabulary
print(len(vocab))

# Convert a tweet into a sequence of word indices.
tweet = 'The movie Pneumonoultramicroscopicsilicovolcanoconiosis is a good movie, it is very funny'
tokens = tokenizer(f'<bos> {tweet} <eos>')
print(tokens)
indices = vocab.forward(tokens)
print(indices)
```

**Task**: What is the index of the `<pad>` token?

```python
# TODO: write code to identify the index of the `<pad>` token
vocab.forward(['<pad>']) # SOLUTION: 3
```

Now let's apply this transformation to the entire set of training, validation, and test data.

```python
def convert_indices(data, vocab):
    """Convert data of form [(tweet, label)...] where tweet is a string
    into an equivalent list, but where the tweets represented as a list
    of word indices.
    """
    return [(vocab.forward(tokenizer(f'<bos> {text} <eos>')), label)
            for (text, label) in data]

train_data_indices = convert_indices(train_data, vocab)
val_data_indices = convert_indices(val_data, vocab)
test_data_indices = convert_indices(test_data, vocab)
```

We have seen that PyTorch's `DataLoader` provides an easy way to form minibatches 
when we worked with image data. However, text and sequence data is more challenging to
work with since the sequences may not be the same length.

Although we can (and will!) continue to use `DataLoader` for our text data, we need to
provide a function that merges sequences of various lengths into two PyTorch tensors 
correspondingg to the inputs and targets for that batch.


**Task**: Following the instructions below, complete the `collate_batch` function,
which creates the input and target tensors
for a batch of data.


```python
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def collate_batch(batch):
    """
    Returns the input and target tensors for a batch of data

    Parameters:
        `batch` - An iterable data structure of tuples (indices, label),
                  where `indices` is a sequence of word indices, and 
                  `label` is either 1 or 0.

    Returns: a tuple `(X, t)`, where 
        - `X` is a PyTorch tensor of shape (batch_size, sequence_length)
        - `t` is a PyTorch tensor of shape (batch_size)
    where `sequence_length` is the length of the longest sequence in the batch
    """

    text_list = []  # collect each sample's sequence of word indices
    label_list = [] # collect each sample's target labels
    for (text_indices, label) in batch:
        text_list.append(torch.tensor(text_indices))
        # TODO: what do we need to do with `label`?
        label_list.append(label) # SOLUTION

    X = pad_sequence(text_list, padding_value=3).transpose(0, 1)
    t = None # TODO
    t = torch.tensor(label_list) # SOLUTION
    return X, t


train_dataloader = DataLoader(train_data_indices, batch_size=10, shuffle=True,
                              collate_fn=collate_batch)
```

With the above code in mind, we should be able to extract batches from `train_dataloader`.
Notice that `X.shape` is different in each batch.
You should also see that the index `3` is used to pad shorter sequences in in a batch.

```python
for i, (X, t) in enumerate(train_dataloader):
    print(X.shape, t.shape)
    if i >= 10:
        break

print(X)
```

**Task**: Why does each sequence begin with the token `0`, and end with the token `1` (ignoring
the paddings).

```python
# TODO: Your explanation goes here
```

## Part 2. Model

We will use a recurrent neural network model to classify positive vs negative
sentiments. Our RNN model will have three components that are typical in a
sequence classification model:

- An *embedding layer*, which will map each word index (akin to a one-hot embedding)
  into a low-dimensional vector. This layer as having the same functionality as the
  weights $W^{(word)}$ from lab 2.
- A *recurrent layer*, which performs the recurrent neural network computation.
  The input to this layer is the low-dimensional embedding vectors
  for each word in the sequence.
- A *fully connected layer*, which computes the final binary classification using
  features computed from the recurrent layer. In our case, we concatenate the
  *max* and *mean* of the hidden units across the time steps (i.e. across each word).

Let's define the model that we will use, and then explore it step by step.

```python
import torch.nn as nn

class MyRNN(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_classes):
        super(MyRNN, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.RNN(emb_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, X):
        # Look up the embedding
        wordemb = self.emb(X)
        # Forward propagate the RNN
        h, out = self.rnn(wordemb)
        # combine the hidden features computed from *each* time step of
        # the RNN. we do this by 
        features = torch.cat([torch.amax(h, dim=1),
                              torch.mean(h, dim=1)], axis=-1)
        # Compute the final prediction
        z = self.fc(features)
        return z

model = MyRNN(len(vocab), 128, 64, 2)
```

To explore exactly what this model is doing, let's grab one batch of data from
the data loader we created. We will observe, step-by-step, what computation will be
performed on the input `X` to obtain the final prediction. We do this by 
emulating the `forward` method of the `MyRNN` function.


```python
X, t = next(iter(train_dataloader))

print(X.shape)
```

**Graded Task**: Run the code below to check the shape of `wordemb`.
What shape does this tensor have?  Explain what each dimension in this shape means.

```python
wordemb = model.emb(X)

print(wordemb.shape)

# TODO: Include your explanation here
```

**Graded Task**: Run the code below, which computes the RNN forward pass,
with `wordemb` as input.
What shape do the tensors `h` and `out` have?  Explain what these tensors correspond to.
(See the RNN reference [https://pytorch.org/docs/stable/generated/torch.nn.RNN.html](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html) on the PyTorch documentation page.)


```python
h, out = model.rnn(wordemb)

print(h.shape)
print(out.shape)

# The tensors `h` and `out` are related. To see the relation,
# choose an index in the batch and compare the following two
# vectors in `h` and `out`.
index = 2 # choose an index to iterate through the batch
print(h[index, -1, :])
print(out[0, index, :])

# TODO: Include your explanation here
```

**Graded Task**: There is a step in the MyRNN forward pass
that combines the features from *each* time step of the RNN by 
computing:

1. the *maximum* value of each position in the hidden vector.
2. the *mean* value of each position in the hidden vector.
3. concatenating the resulting two vectors.

(Note that in the demo below, we are working with a minibatch. Thus,
each of `out1`, `out2`, and `features` below are *matrices* containing
the vectors from each minibatch)

This method typically performs better than, say, taking the hidden
state at the last time step (the value `out` from above). Explain,
intuitively, why you might expect this performance to be the case for a
sentiment analysis task.

```python
out1 = torch.amax(h, dim=1)
out2 = torch.mean(h, dim=1)
features = torch.cat([out1, out2], axis=-1)

# Compare, for a single input in the batch, the connection between
# `h`, `out1`, `out2` and `features`:
print(h[index, :, :])
print(out1[index, :])
print(out2[index, :])
print(features[index, :])

# TODO: Include your explanation here
```

**Task**: Finally, the model uses the `features` tensor to compute
the prediction for each element in the batch. Run the code below to
complete this step.

```python
print(model.fc(features))
```

There is one more thing we need to do before training the model, which is
to write a function to estimate the accuracy of the model. This is done for 
you below.


```python
def accuracy(model, dataset, max=1000):
    """
    Estimate the accuracy of `model` over the `dataset`.
    We will take the **most probable class**
    as the class predicted by the model.

    Parameters:
        `model`   - An object of class nn.Module
        `dataset` - A dataset of the same type as `train_data`.
        `max`     - The max number of samples to use to estimate 
                    model accuracy

    Returns: a floating-point value between 0 and 1.
    """

    correct, total = 0, 0
    dataloader = DataLoader(dataset,
                            batch_size=1,  # use batch size 1 to prevent padding
                            collate_fn=collate_batch)
    for i, (x, t) in enumerate(dataloader):
        z = model(x)
        y = torch.argmax(z, axis=1)
        correct += int(torch.sum(t == y))
        total   += 1
        if i >= max:
            break
    return correct / total

accuracy(model, train_data_indices) # should be close to half
```

## Part 3. Training

In this section, we will train the `MyRNN` model to classify tweets.
As the models that we are building begin to increase in complexity, it is important
to use good debugging techniques. In this section, we will introduce the technique of
checking whether the model and training code is able to overfit on a small training set.
This is a way to check for bugs in the implementation.

**Task**: Complete the training code below


```python
import torch.optim as optim 
import matplotlib.pyplot as plt

def train_model(model,                # an instance of MLPModel
                train_data,           # training data
                val_data,             # validation data
                learning_rate=0.001,
                batch_size=100,
                num_epochs=10,
                plot_every=50,        # how often (in # iterations) to track metrics
                plot=True):           # whether to plot the training curve
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               collate_fn=collate_batch,
                                               shuffle=True) # reshuffle minibatches every epoch
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # these lists will be used to track the training progress
    # and to plot the training curve
    iters, train_loss, train_acc, val_acc = [], [], [], []
    iter_count = 0 # count the number of iterations that has passed

    try:
        for e in range(num_epochs):
            for i, (texts, labels) in enumerate(train_loader):
                z = None # TODO
                z = model(texts) # SOLUTION

                loss = None # TODO
                loss = criterion(z, labels) # SOLUTION

                loss.backward() # propagate the gradients
                optimizer.step() # update the parameters
                optimizer.zero_grad() # clean up accumualted gradients

                iter_count += 1
                if iter_count % plot_every == 0:
                    iters.append(iter_count)
                    ta = accuracy(model, train_data)
                    va = accuracy(model, val_data)
                    train_loss.append(float(loss))
                    train_acc.append(ta)
                    val_acc.append(va)
                    print(iter_count, "Loss:", float(loss), "Train Acc:", ta, "Val Acc:", va)
    finally:
        # This try/finally block is to display the training curve
        # even if training is interrupted
        if plot:
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
            plt.ylabel("Loss")
            plt.legend(["Train", "Validation"])
```

**Graded Task**: As a way to check the model and training code, 
check if your model can obtain a 100\% training accuracy relatively quickly
(e.g. within less than a minute of training time), when training on only the
first 20 element of the training data.


```python
model = MyRNN(vocab_size=len(vocab),
              emb_size=300,
              hidden_size=64,
              num_classes=2)
# TODO: Include your code and output 
train_model(model, train_data_indices[:20], val_data_indices[:20], # SOLUTION
            batch_size=10, num_epochs=100, plot_every=1,    # SOLUTION
            learning_rate=0.001)                            # SOLUTION
```

**Task**: Will this model that you trained above have a high accuracy over
the validation set? Explain why or why not.

```python
# TODO: Your explanation goes here
```

**Graded Task**: Train your model on the full data set. What validation accuracy
can you achieve?

```python
# TODO: Include your code here. Try a few hyperparameter choices until you
# are satisfied that your model performance is reasonable (i.e. no obviously
# poor hyperparameter choices)
model = MyRNN(vocab_size=len(vocab), emb_size=300, hidden_size=64, num_classes=2) # SOLUTION
train_model(model, train_data_indices, val_data_indices, batch_size=100, num_epochs=20, learning_rate=0.001) # SOLUTION
```

Instead of a (vanilla) RNN model, PyTorch also makes available 
`nn.LSTM` and `nn.GRU` units. They can be used in place of `nn.RNN` without 
further changes to the `MyRNN` code.

In general, gated units like LSTM's are much more frequently used than vanilla RNNs,
although transformers are much more popular now as well.

## Part 4. Pretrained Embeddings

As we saw in the previous lab on images, **transfer learning** is a useful technique
in practical machine learning, especially in low-data settings:
instead of training an entire neural network from scratch, we use (part of) a
model that is pretrained on large amounts of similar data. We use the intermediate
state of this pretrained model as features to our model---i.e. we use the pretrained
models to compute *features*.

Just like with images, using a pretrained model is an important strategy for working
with text. Large language models is an excellent demonstration of how generalizable
pretrained features can be.

In this part of the lab, we will use a slightly older idea of using pretrained *word embeddings*.
In particular, instead of training our own `nn.Embedding` layer, we will use
GloVe embeddings (2014) [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)
trained on a large data set containing all of Wikipedia and other webpages.

Nowadays, large language model (LLMs), including those with APIs provided by various organizations,
can also be used to map words/sentences into embeddings.
However, the basic idea of using pretrained models in low-data settings remains similar.
We will also identify some bias issues with pretrained word embeddings.
There is evidence that these types of bias issues 
continues to persist in LLMs as well.


```python
from torchtext.vocab import GloVe

glove = torchtext.vocab.GloVe(name="6B", dim=300)
```

**Task**: Run the below code to print the GloVe word embedding for the word "cat".

```python
print(glove['cat'])
```

Unfortunately, it is not straightforward to add the `<pad>`, `<unk>`, `<bos>` and `<eos>`
tokens. So we will do without them.

**Task**: Run the below code to look up GloVe word indices for the training, validation, and 
test sets.

```python
def convert_indices_glove(data, default=len(glove)-1):
    result = []
    for text, label in data:
        words = tokenizer(text) # for simplicity, we wont use <bos> and <eos>
        indices = []
        for w in words:
            if w in glove.stoi:
                indices.append(glove.stoi[w])
            else:
                # this is a bit of a hack, but we will repurpose *last* word
                # (least common word) appearing in the GloVe vocabluary as our
                # '<pad>' token
                indices.append(default)
        result.append((indices, label),)
    return result

train_data_glove = convert_indices_glove(train_data)
val_data_glove = convert_indices_glove(val_data)
test_data_glove = convert_indices_glove(test_data)
```

Now, we will modify the `MyRNN` to use the pretrained GloVe vectors:

```python
class MyGloveRNN(nn.Module):
    def __init__(self,  hidden_size, num_classes):
        super(MyGloveRNN, self).__init__()
        self.vocab_size, self.emb_size = glove.vectors.shape
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.emb = nn.Embedding.from_pretrained(glove.vectors)
        self.emb.requires_grad=False # do *not* update the glove embeddings
        self.rnn = nn.RNN(self.emb_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, X):
        # Look up the embedding
        wordemb = self.emb(X)
        # Forward propagate the RNN
        h, out = self.rnn(wordemb)
        # combine the hidden features computed from *each* time step of
        # the RNN. we do this by 
        features = torch.cat([torch.amax(h, dim=1),
                              torch.mean(h, dim=1)], axis=-1)
        # Compute the final prediction
        z = self.fc(features)
        return z

    def parameters(self):
        # do not return the parameters of self.emb 
        # so the optimizer will not update the parameters of self.emb
        return (p for p in super(MyGloveRNN, self).parameters() if p.requires_grad)


model = MyGloveRNN(64, 2)
```

**Task** Train this model. Use comparable hyperparameters so that you can compare
your result against `MyRNN`. 

```python
# TODO: Train your model here, and include the output
model = MyGloveRNN(100, 2)                           # SOLUTION
train_model(model, train_data_glove, val_data_glove, # SOLUTION
              batch_size=100, # SOLUTION
              num_epochs=200, # SOLUTION
              plot_every=100, # SOLUTION
              learning_rate=0.001) # SOLUTION
```

**Graded Task**: You might notice that a *very* smaller number of 
iterations will be required to train this model to a reasonable
performance (e.g. >70% validation accuracy). Why might this be?

```python
# TODO: Include your explanation here
```

**Graded Task**: Train both MyGloveRNN and MyRNN models using the corresponding
embeddings (pretrained vs. not), **but only with the first 200 data points in the
training set**. How do the validation accuracies compare between these two models?

```python
# TODO: Training code for MyGloveRNN.
# Include outputs and training curves in your submission
glove_model = MyGloveRNN(100, 2)                                  # SOLUTION
train_model(glove_model, train_data_glove[:200], val_data_glove, # SOLUTION
              batch_size=200, # SOLUTION
              num_epochs=200, # SOLUTION
              plot_every=100, # SOLUTION
              learning_rate=0.001) # SOLUTION
```


```python
# TODO: Training code for MyRNN
# Include outputs and training curves in your submission
rnn_model = MyRNN(len(vocab), 300, 100, 2)                                  # SOLUTION
train_model(rnn_model, train_data_indices[:200], val_data_indices, # SOLUTION
              batch_size=200, # SOLUTION
              num_epochs=200, # SOLUTION
              plot_every=100, # SOLUTION
              learning_rate=0.001) # SOLUTION
```
```python
# TODO: Compare the validation accuaries here
```


Machine learning models have an air of "fairness" about them, since models
make decisions without human intervention. However, models can and do learn
whatever bias is present in the training data.
GloVe vectors seems innocuous enough: they are just representations of
words in some embedding space. Even so, we will show that the structure
of the GloVe vectors encodes the everyday biases present in the texts
that they are trained on.

We start with an example analogy to demonstrate the power of GloVe embeddings
that allows us to complete analogies by applying arithmetic operations
to the word vectors.

$$doctor - man + woman \approx ??$$

To find the answers to the above analogy, we will compute the following vector,
and then find the word whose vector representation is *closest* to it.

```python
v = glove['doctor'] - glove['man'] + glove['woman']
```

**Task**: Run the code below to find the closets word. You should see the word
"nurse" fairly high up in that list.

```python
def print_closest_words(vec, n=5):
    dists = torch.norm(glove.vectors - vec, dim=1)     # compute distances to all words
    lst = sorted(enumerate(dists.numpy()), key=lambda x: x[1]) # sort by distance
    for idx, difference in lst[1:n+1]: 					       # take the top n
        print(glove.itos[idx], difference)

print_closest_words(v)
```

**Task**:  To compare, use a similar method to find the answer to this analogy:
$$doctor - woman + man \approx ??$$

In other words, we go the opposite direction in the "gender" axis to check
if similarly concerning analogies exist.

```python
print_closest_words(glove['doctor'] - glove['woman'] + glove['man'])
```


**Task**: Compare the following two outputs.

```python
print_closest_words(glove['programmer'] - glove['man'] + glove['woman'])
```

```python
print_closest_words(glove['programmer'] - glove['woman'] + glove['man'])
```

**Task**: Compare the following two outputs.

```python
print_closest_words(glove['professor'] - glove['man'] + glove['woman'])
```

```python
print_closest_words(glove['professor'] - glove['woman'] + glove['man'])
```

**Task**: Compare the following two outputs.

```python
print_closest_words(glove['engineer'] - glove['man'] + glove['woman'])
```

```python
print_closest_words(glove['engineer'] - glove['woman'] + glove['man'])
```

**Graded Task**: Explain where the bias in these embeddings come from.
Would you expect our word embeddings (trained on tweets) to be similarly
problematic? Why or why not?

```python
# TODO: Your explanation goes here
```

