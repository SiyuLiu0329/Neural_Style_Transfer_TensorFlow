# Neural style transfer in TensorFlow

### Example 1
<img src="demo/style1.jpg" width="400"><img src="demo/content1.jpg" width="400">
<p>
<img src="demo/res1.png" width="800">

### Example 2
<img src="demo/style2.jpg" width="400"><img src="demo/content2.jpg" width="400">
<p>
<img src="demo/res2.png" width="800">

### Example 3
<img src="demo/style3.jpg" width="400"><img src="demo/content3.jpg" width="400">
<p>
<img src="demo/res3.png" width="800">

### Example 4
<img src="demo/style4.jpg" width="400"><img src="demo/content4.jpg" width="400">
<p>
<img src="demo/res4.png" width="800">

### Example 5
<img src="demo/style5.jpg" width="400"><img src="demo/content5.jpg" width="400">
<p>
<img src="demo/res5.png" width="800">

# About This Project
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

- Resize the input image by padding or cropping, pass in a new size *shape=[height, width]*. **(Black bars as a result of padding will affect the output image.)**

- Convert the input image to BGR format by passing in *bgr=True* - this conversion is required by some VGG models. Does not seem to make a difference with the weights used in the projects.

- Show the output image in a new window by passing in *show_img=True* - Cannot be used with image preprocessing as negative-valued pixels will not be displayed properly.

- Preprocess the input image by passing in *preprocess=True* - subtract the vgg-means from the RGB channels of the input image.

# Note
The model only trains on CPUs since I only own a Mac. In most cases 50 (30 min) iterations can output suprisingly good results for the time given it is given. The model will improve very slowly from this point on.

Iter = 10 (Left) and iter = 20 (Right).
<p>
    <img src="demo/10.png" width="400"><img src="demo/20.png" width="400">
<p>
Iter = 30 (Left) and iter = 40 (Right).
<p>
    <img src="demo/30.png" width="400"><img 
    src="demo/40.png" width="400">
<p>
Iter = 50 (Left) and iter = 200 (Right).
<p>
    <img src="demo/50.png" width="400"><img 
    src="demo/200.png" width="400">
<p>
Iter = 460 (Left) and iter = 1110 (Right).
<p>
    <img src="demo/460.png" width="400"><img src="demo/1110.png" width="400">
<p>


# TODO
- TensorFlow Saver
- More effective image resizing function
- Possible GPU version

