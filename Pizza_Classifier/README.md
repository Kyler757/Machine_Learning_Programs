# Image classifying pizza
This project uses a neural network to classify if an image has a pizza in it or not. To accomplish this the neural network uses multiple convolutional layers to shrink down the image into the output of the classifier, a single number. This number is between 0 and 1. When a number closer to 1 is given by the classifier, then the image is classified as pizza, otherwise it is not.
## Training
The classifier was given 384 X 512 images of pizza and images not of pizza and was asked to classify them. Using Binary cross entropy loss, the model was trained based on how well it classified the images. The dataset is from [Kaggle](https://www.kaggle.com/datasets/carlosrunner/pizza-not-pizza)
## Model
To train or run the model utilize the `trainNN` function. The `trainNN` function has the following parameters.
- `epochs` - Set to 0 to run the model without training. Every epoch represents one pass through the dataset.
- `batch_size` - The number of images trained the model on at a time.
- `lr` - How much the model parameters are adjusted after each iteration.
- `save_time` - The number of epochs before each save.
- `save_dir` - The file where the model is saved and loaded from.

