---
type: ProjectLayout
title: Galaxy Classification
colors: colors-a
date: '2021-10-15'
client: ''
description: >-
  JAX-based deep net to distinguish galaxy types based on images from the Sloan
  Digital Sky Survey (SDSS) dataset.
featuredImage:
  type: ImageBlock
  url: /images/Screenshot 2025-06-01 031604.png
  altText: Project thumbnail image
media:
  type: ImageBlock
  url: /images/Screenshot 2025-06-01 031604.png
  altText: Project image
---
This project is a neural network built using Python's JAX library to classify galaxies from the Sloan Digital Sky Survey (SDSS) into categories such as spiral, elliptical, or irregular. The model takes in preprocessed galaxy images and learns to distinguish between morphological types using a two-layer architecture optimized for vectorized operations.

It implements a custom training loop using cross-entropy loss minimization and JAX gradient descent update functions for accelerated performance. This two-layer architecture proved useful in distinguishing between and classifying images of galaxies from the SDSS dataset (out of five possible morphological types) through label classification.

### Part 1: The Dataset

![](/images/Screenshot%202025-06-03%20222318.png)

The types were organized into these categories:

**Galaxy10 dataset (25753 images)**

*   **0** Class 0 (3461 images): Disk, Face-on, No Spiral

*   **1** Class 1 (6997 images): Smooth, Completely round

*   **2** Class 2 (6292 images): Smooth, in-between round

    *   Class 3 (394 images): Smooth, Cigar shaped

*   **3** Class 4 (3060 images): Disk, Edge-on, Rounded Bulge

    *   Class 5 (17 images): Disk, Edge-on, Boxy Bulge

    *   Class 6 (1089 images): Disk, Edge-on, No Bulge

*   **4** Class 7 (1932 images): Disk, Face-on, Tight Spiral

    *   Class 8 (1466 images): Disk, Face-on, Medium Spiral

    *   Class 9 (1045 images): Disk, Face-on, Loose Spiral

For example, a label of \[0, 1, 0, 0, 0] (single channel, but original dataset has three channels with g, r, i labels which can be used) corresponds to a smooth, completely round galaxy. An example of a disk, face-on, no spiral galaxy (\[1, 0, 0, 0, 0]) is shown above (the first image in this page).

The neural network, which took in our input image and outputted a size-5 vector, is described as follows:

![](/images/Screenshot%202025-06-03%20223315.png)

where V<sub>in</sub> has size (17457, 4761) and v<sub>out</sub> has size (5, 1).

Two measures of goodness were used. The first one is minimization of cross-entropy loss:

![](/images/Screenshot%202025-06-03%20223615.png)

```
def loss(params, imageVector, correctLabel):
    a = net(params, imageVector)
    sum = 0
    for i in range(len(correctLabel)):
        sum += -correctLabel[i] * jnp.log(a[i]) - (1 - correctLabel[i]) * jnp.log(1 - a[i])
    return sum
```



The second evaluation of performance is the fraction of labels correct for each image in our training set (if in our vector, the maximum value is where the 1 should be).



### Part 2: Training Loop

The training loop was non-complex, where I simply chose a random image in the training set, computed the gradient (through JAX's `grad(loss)` function), and observed how the fraction of correct labels changed for 100,000 steps.

```
loss_grad = jit(jax.grad(loss))
fraction_list = []

for i in range(50 * 2000):
    if i % 1000 == 0:
        print(f"Iteration {i}")        
        fraction_list.append(fractionCorrect(params,
            test_images, test_labels))
        print(f"Fraction correct on test set:
            {fraction_list[-1]}")
    idx = np.random.randint(len(train_images))
    grad = loss_grad(params, train_images[idx],
            train_labels[idx])
    for j in range(len(params)):
        params[j] -= 0.001 * grad[j]
```



The final fraction of labels correct on the test set after all iterations was 0.742. The results are as follows:

![](/images/Screenshot%202025-06-01%20213253.png)

As you can see, the accuracy oscillates between 0.740 and 0.755. However, I found a way to not only increase this accuracy, but also reduce the time this neural net takes to train on the data.

Using `jax.vmap`, I was able to make significant performance improvements (additionally, remove an extra for-loop and write a faster "fraction correct" function):

![](/images/Screenshot%202025-06-01%20222032.png)

with the final fraction of correct labels being 0.755. Although the accuracy is not heavily increased, the training time was cut down nearly tenfold, allowing for future augmentations to the neural network and processing of much larger datasets with minimal time delays.



### For code and dataset:

<https://github.com/siddhshah/GalaxyClassification>

