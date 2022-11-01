"""
The model builder to build different semantic segmentation models.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
from models import *
from models.pidnet import get_pred_model, PIDNet
import tensorflow as tf

layers = tf.keras.layers


def builder(num_classes, input_size=(256, 256), model='SegNet', base_model=None):
    models = {'FCN-8s': FCN,
              'FCN-16s': FCN,
              'FCN-32s': FCN,
              'UNet': UNet,
              'SegNet': SegNet,
              'Bayesian-SegNet': SegNet,
              'PAN': PAN,
              'PSPNet': PSPNet,
              'RefineNet': RefineNet,
              'DenseASPP': DenseASPP,
              'DeepLabV3': DeepLabV3,
              'DeepLabV3Plus': DeepLabV3Plus,
              'BiSegNet': BiSegNet,
              'PIDNet':PIDNet}

    assert model in models

    if 'PIDNet' == model:
        model_name = "pidnet_m"
        net = get_pred_model(model_name,input_size+(3,),num_classes)

        return net, "resnet"

    else:
        net = models[model](num_classes, model, base_model)
        inputs = layers.Input(shape=input_size+(3,))

        return net(inputs), net.get_base_model()
