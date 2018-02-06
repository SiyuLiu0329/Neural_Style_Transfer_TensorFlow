# Neural style transfer in TensorFlow
A TensorFlow implementation of [neural style transfer](https://arxiv.org/pdf/1508.06576.pdf) with [VGG19](https://arxiv.org/pdf/1409.1556.pdf) (a [working version](https://github.com/SiyuLiu0329/Neural_Style_Transfer_TensorFlow/blob/master/vgg_model.py) of VGG19 is also included). This implementation is inspired by a deep learning course developed by deeplearning.ai. The default hyper parameters used in the [model file](https://github.com/SiyuLiu0329/Neural_Style_Transfer_TensorFlow/blob/master/neural_style.py) have been fine tuned with the help of this great [paper](https://arxiv.org/pdf/1705.04058.pdf) so the model should be able produce satisfactory results with most style-content combinations.

# Files Included
-----------
```
neural_style.py
test_neural_style.py
test_vgg.py
utilities.py
vgg_model.py

README.md
```

# Files Required but Not Included
You need to download the following files and place them in the same directory as the model in order to run the model

- [imagenet-vgg-verydeep-19.mat](http://www.vlfeat.org/matconvnet/pretrained/)

- [synset.txt](https://github.com/machrisaa/tensorflow-vgg/blob/master/synset.txt)

# TODO: Usage


