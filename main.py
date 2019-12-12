# try:
#   # %tensorflow_version only exists in Colab.
#   %tensorflow_version 2.x
# except Exception:
#   pass
import tensorflow as tf
import numpy as np
import argparse
from __future__ import print_function, division
from tensorflow.python.platform import flags

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from simple_model import DCGAN

import matplotlib.pyplot as plt

import sys


FLAGS = flags.FLAGS
parser = argparse.ArgumentParser(description='Keras GAN example')
parser.add_argument("--epochs", type=int, default=25,
                    help="Epoch to train [25]", required=False)
parser.add_argument("--batch_size", type=int, default=32,
                    help="training batch size", required=False)
parser.add_argument("--discriminator_loss", default='binary_crossentropy',
                    type=str, help="the type of discriminator loss function",
                    required=False)
parser.add_argument("--generator_loss", default='binary_crossentropy',
                    type=str, help="the type of discriminator loss function",
                    required=False)
parser.add_argument("--generator_activation", type=str, default='relu',
                    help="generator activation", required=False)
parser.add_argument("--discriminator_activation", type=str, default='sigmoid',
                    help="generator activation", required=False)
parser.add_argument("--img_row", type=int, default=28,
                    help="image row size, 28 for MNIST", required=False)
parser.add_argument("--img_col", type=int, default=28,
                    help="# usually the same as ROW size", required=False)
parser.add_argument("--learning_rate", type=int, default=0.0002,
                    help=" learning rate for Adam optimizer",
                    required=False)
parser.add_argument("--beta1", type=int, default=0.5,
                    help="Beta1 for Adam optimizer", required=False)
parser.add_argument("--img_color_size", type=int, default=1,
                    help="3 for RGB, 1 for greyscale", required=False)
parser.add_argument("--z_dim", type=int, default=100,
                    help="dimension for the latent image G takes as input",
                    required=False)
args = parser.parse_args(args=[])



def main():
    args = parser.parse_args(args=[])
    epochs = args.epochs
    dcgan = DCGAN(args.img_row, args.img_col, args.img_color_size, args.z_dim,
                  args.learning_rate, args.beta1, args.discriminator_loss,
                  args.generator_loss, args.discriminator_activation,
                  args.generator_activation)
    dcgan.train(epochs=epochs, batch_size=args.batch_size, save_imgs=False)


if __name__ == '__main__':
    parse_args()
    main()

