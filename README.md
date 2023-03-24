# Image Seeker

You write a description, Image Seeker gives you the image that fits the best with it. The image will come from a 1000-images-large dataset (cf. *2. Data*).


###### 1. Anaconda environment

To set up the environment, execute in a terminal the followings:

* `conda create -n clip`
* `conda activate clip`
* `conda install pip`
* `pip install -r requirements-dev.txt`


###### 2. Data

``imagenet-sample``, a small sample of images from the ImageNet 2012 dataset. The dataset contains 1,000 images, one randomly chosen from each class of the validation split of the ImageNet 2012 dataset.


###### 3. Model

The model used in this package is [CLIP](https://openai.com/research/clip), proposed in [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) by Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever. CLIP (Contrastive Language-Image Pre-Training) is a zero-shot neural network trained on a variety of (image, text) pairs.
The version implemented here is `clip-vit-base-patch32`. The model uses a ViT-B/32 Transformer architecture as an image encoder and uses a masked self-attention Transformer as a text encoder. These encoders are trained to maximize the similarity of (image, text) pairs via a contrastive loss.
