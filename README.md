# Neural style transfer in TensorFlow
A TensorFlow implementation of [neural style transfer](https://arxiv.org/pdf/1508.06576.pdf) with [VGG19](https://arxiv.org/pdf/1409.1556.pdf) (a [working version](https://github.com/SiyuLiu0329/Neural_Style_Transfer_TensorFlow/blob/master/vgg_model.py) of VGG19 is also included). This implementation is inspired by a deep learning course developed by [deeplearning.ai](https://www.deeplearning.ai). The default hyper parameters used in the [model file](https://github.com/SiyuLiu0329/Neural_Style_Transfer_TensorFlow/blob/master/neural_style.py) have been fine tuned with the help of this great [paper](https://arxiv.org/pdf/1705.04058.pdf) so the model should be able produce satisfactory results with most style-content combinations.


# Files Included
- [neural_style.py](https://github.com/SiyuLiu0329/Neural_Style_Transfer_TensorFlow/blob/master/neural_style.py)
- [test_neural_style.py](https://github.com/SiyuLiu0329/Neural_Style_Transfer_TensorFlow/blob/master/test_neural_style.py)
- [test_vgg.py](https://github.com/SiyuLiu0329/Neural_Style_Transfer_TensorFlow/blob/master/test_vgg.py)
- [utilities.py](https://github.com/SiyuLiu0329/Neural_Style_Transfer_TensorFlow/blob/master/utilities.py)
- [vgg_model.py](https://github.com/SiyuLiu0329/Neural_Style_Transfer_TensorFlow/blob/master/vgg_model.py)
- [README.md](https://github.com/SiyuLiu0329/Neural_Style_Transfer_TensorFlow/blob/master/README.md)


# Files Required but Not Included
Download the following files and place them in the same directory to run the model
- [imagenet-vgg-verydeep-19.mat](http://www.vlfeat.org/matconvnet/pretrained/) - pretrained weights for VGG19
- [synset.txt](https://github.com/machrisaa/tensorflow-vgg/blob/master/synset.txt) - *(Optional)* download only if you want to make predictions with the included VGG19 model.

# Usage
## Using the model
To use the model for art style transferring, simply run the following code
```
model = NSTModel(content_path='content_img_path', style_path='style_img_path')
model.run()
```

The default image size is 224*224. Feel free to try out other sizes, make sure that the content image and the style image are the same size.

## Loading and preprocessing images
A (kind of inefficent) image loading and preprocessing function *(load_image)* is provided in [utilities.py](https://github.com/SiyuLiu0329/Neural_Style_Transfer_TensorFlow/blob/master/utilities.py). Use this function to

- Resizing the input image by padding or cropping, pass in a new size *shape=[height, width]*. Resizing is NOT recommended!

- Convert the input image to BGR format by passing in *bgr=True* - this conversion is required by some VGG models. Does not seem to make a difference with the weights used in the projects.

- Show the output image in a new window by passing in *show_img=True* - Cannot be used with image preprocessing as negative-valued pixels will not be displayed properly.

- Preprocess the input image by passing in *preprocess=True* - subtract the vgg-means from the RGB channels of the input image.


# TODO
- TensorFlow Saver
- More effective image resizing function

