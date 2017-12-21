# Face-generating GAN #
Generative Adversarial Network designed to produce realistic human faces after being trained on a sample of 200,000+ images of human faces

### What is this repository for? ###

* This repository is for the Jupyter notebook and supporting python files
* The data files used for this project can be found either in my Bitbucket (as it was too much for my Github) or at the websites for the [MNIST](http://yann.lecun.com/exdb/mnist/) and [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) datasets.
* Version: stable

### How do I get set up? ###

* To set up, clone the repository or download a .zip file with the repository to your desired directory. Make sure that the directory contains a folder titled "data" to include the MNIST and CelebA folders in.
* Configuration
* The helper.py and unittests.py files are dependencies that should also be included in the directory
* Unit tests are set up to automatically be run by the python notebook before the running of the Genrative Adversarial Network.
* To start, run the Jupyter Notebook application in an environment (such as that created by Anaconda) for 

### Contribution guidelines ###

* This repository can easily be forked and contributed to on Github. Feel free to send a pull request.

### Future Steps ###

* The current GAN makes realistic enough faces, but further hyperparameter optimization is needed. 
* [DRAW: A Recurrent Neural Network For Image Generation](https://arxiv.org/pdf/1502.04623.pdf) has demonstrated promising results on the  MNIST generation task. Next step will involve applying this model to the CelebA dataset.
* Refactoring the code to make use of Google Brain's [TFGAN](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/gan) library will [make future modifications easier](https://research.googleblog.com/2017/12/tfgan-lightweight-library-for.html), as well as [provide more useful metrics of performance](https://datahub.packtpub.com/deep-learning/google-opensources-tensorflow-gan-tfgan-library-for-generative-adversarial-networks-neural-network-model/).
