"""
The implementation of some losses based on Tensorflow.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
import tensorflow as tf
from keras import backend as K

backend = tf.keras.backend

# pytorch에서 reduction 방법을 mean으로 함
def categorical_crossentropy_with_logits(y_true, y_pred):
    # compute cross entropy
    cross_entropy = backend.categorical_crossentropy(y_true, y_pred, from_logits=True)

    # compute loss
    loss = backend.mean(backend.mean(cross_entropy, axis=[1, 2]))
    return loss

# pytorch에서 reduction 방법을 mean으로 함
def focal_loss(alpha=0.25, gamma=4.0):
    def loss(y_true, y_pred):
        y_pred = backend.softmax(y_pred,axis=-1)

        # compute ce loss
        cross_entropy = backend.categorical_crossentropy(y_true, y_pred, from_logits=False)

        # compute weights
        weights = backend.sum(alpha * backend.pow(1 - y_pred, gamma) * y_true, axis=-1)
        return backend.mean(backend.mean(weights * cross_entropy, axis=[1, 2]))

    return loss

def miou_loss(weights=None, num_classes=2):
    if weights is not None:
        assert len(weights) == num_classes
        weights = tf.convert_to_tensor(weights)
    else:
        weights = tf.convert_to_tensor([1.] * num_classes)

    def loss(y_true, y_pred):
        y_pred = backend.softmax(y_pred)

        inter = y_pred * y_true
        inter = backend.sum(inter, axis=[1, 2])

        union = y_pred + y_true - (y_pred * y_true)
        union = backend.sum(union, axis=[1, 2])

        return -backend.mean((weights * inter) / (weights * union + 1e-8))

    return loss


def self_balanced_focal_loss(alpha=3, gamma=2.0):
    """
    Original by Yang Lu:

    This is an improvement of Focal Loss, which has solved the problem
    that the factor in Focal Loss failed in semantic segmentation.
    It can adaptively adjust the weights of different classes in semantic segmentation
    without introducing extra supervised information.

    :param alpha: The factor to balance different classes in semantic segmentation.
    :param gamma: The factor to balance different samples in semantic segmentation.
    :return:
    """

    def loss(y_true, y_pred):
        # cross entropy loss
        y_pred = backend.softmax(y_pred, -1)
        cross_entropy = backend.categorical_crossentropy(y_true, y_pred)

        # sample weights
        sample_weights = backend.max(backend.pow(1.0 - y_pred, gamma) * y_true, axis=-1)

        # class weights
        pixel_rate = backend.sum(y_true, axis=[1, 2], keepdims=True) / backend.sum(backend.ones_like(y_true),
                                                                                   axis=[1, 2], keepdims=True)
        class_weights = backend.max(backend.pow(backend.ones_like(y_true) * alpha, pixel_rate) * y_true, axis=-1)

        # final loss
        final_loss = class_weights * sample_weights * cross_entropy
        return backend.mean(backend.mean(final_loss, axis=[1, 2]))

    return loss

def dice_coef(y_true,y_pred,smooth=1e-6):
    dice = 0

    # 배치사이즈만큼 돈다.
    for i in range(y_true.shape[0]):
        y_true_f = K.flatten(y_true[i])
        y_pred_f = K.flatten(y_pred[i])

        intersection = K.sum(y_true_f*y_pred_f)
        dice += (2.*intersection+smooth)/(K.sum(y_true_f)+K.sum(y_pred_f)+smooth)

    return dice / y_true.shape[0]

def dice_coef_loss(smooth=1e-6):

    def loss(y_true,y_pred):
        return 1-dice_coef(y_true,y_pred,smooth)

    return loss

