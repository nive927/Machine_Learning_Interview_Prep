# Interview Ready Guide

## Overfitting

<details>
<summary>What are some approaches to reducing overfitting in machine learning models? List them in order of preference and explain why regularization is not always the first choice?</summary>
 
We do not necessarily need to rely on dropout or other regularization approaches to reduce overfitting. There are other techniques we should try first, since regularization, by definition, biases our model towards simplicity—which we only want to do if we know that's necessary. This is the order that we recommend using for reducing overfitting (more details about each in a moment):

- Add more data
- Use data augmentation
- Use architectures that generalize well
- Add regularization (e.g. dropout, l1/l2 regularization)
- Reduce architecture complexity (reduce filters → hard to do)

Assume you've already collected as much data as you can, so step (1) isn't relevant (this is true for most Kaggle competitions, for instance). The next step (2) is data augmentation. This refers to creating additional synthetic data, based on reasonable modifications of your input data. For images, this is likely to involve one or more of: flipping, rotation, zooming, cropping, panning, minor color changes.

Which types of augmentation are appropriate depends on your data. For regular photos, for instance, you'll want to use horizontal flipping, but not vertical flipping (since an upside down car is much less common than a car the right way up, for instance!).

It is recommended to always use at least some light data augmentation, unless you have so much data that your model will never see the same input twice.

</details>

## Debugging and Observing Model Predictions

<details>
<summary>How would you debug and observe model predictions to better understand your model's performance? Provide code examples for each step.</summary>

First, calculate the predictions on the validation set, since we know those labels, rather than looking at the test set.

```python
vgg.model.load_weights(latest_weights_filename)
```

```python
val_batches, probs = vgg.test(VAL_PATH, batch_size = batch_size)
# Found 2000 images belonging to 2 classes.
```

```python
filenames = val_batches.filenames
expected_labels = val_batches.classes # makes them 0 or 1

our_predictions = probs[:, 0]
our_labels = np.round(1 - our_predictions)
```

**Ways to observe predictions:**

1. **A few correct labels at random**
    ```python
    correct = np.where(preds==val_labels[:,1])[0]
    idx = permutation(correct)[:n_view]
    plots_idx(idx, probs[idx])
    ```

2. **A few incorrect labels at random**
    ```python
    incorrect = np.where(preds!=val_labels[:,1])[0]
    idx = permutation(incorrect)[:n_view]
    plots_idx(idx, probs[idx])
    ```

3. **The most correct labels of each class (highest probability that are correct)**
    ```python
    correct_cats = np.where((preds==0) & (preds==val_labels[:,1]))[0]
    most_correct_cats = np.argsort(probs[correct_cats])[::-1][:n_view]
    plots_idx(correct_cats[most_correct_cats], probs[correct_cats][most_correct_cats])
    ```

4. **The most incorrect labels of each class (highest probability that are incorrect)**
    ```python
    incorrect_dogs = np.where((preds==1) & (preds!=val_labels[:,1]))[0]
    most_incorrect_dogs = np.argsort(probs[incorrect_dogs])[:n_view]
    plots_idx(incorrect_dogs[most_incorrect_dogs], 1-probs[incorrect_dogs][most_incorrect_dogs])
    ```

5. **The most uncertain labels (probability closest to 0.5)**
    ```python
    most_uncertain = np.argsort(np.abs(probs-0.5))
    plots_idx(most_uncertain[:n_view], probs[most_uncertain])
    ```

_Source: https://www.cs.utah.edu/~cmertin/dogs+cats+redux.html_

</details>