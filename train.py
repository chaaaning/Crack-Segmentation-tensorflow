from utils.data_generator import ImageDataGenerator
from utils.helpers import get_dataset_info, check_related_path
from utils.callbacks import LearningRateScheduler
from utils.optimizers import *
from utils.losses import *
from utils.learning_rate import *
from utils.metrics import MeanIoU, F1Score
from utils import utils
from builders import builder
from tqdm import tqdm
import logging

from tensorflow.keras.optimizers.schedules import CosineDecay,PolynomialDecay,ExponentialDecay
import tensorflow_model_optimization as tfmot
import tensorflow as tf
import argparse
import os
import wandb
from eval import evaluate


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Choose the semantic segmentation methods.', type=str, required=True)
parser.add_argument('--base_model', help='Choose the backbone model.', type=str, default=None)
parser.add_argument('--dataset', help='The path of the dataset.', type=str, default='CamVid')
parser.add_argument('--loss', help='The loss function for traing.', type=str, default=None,
                    choices=['ce', 'focal_loss', 'miou_loss', 'self_balanced_focal_loss','dice_loss'])
parser.add_argument('--num_classes', help='The number of classes to be segmented.', type=int, default=32)
parser.add_argument('--random_crop', help='Whether to randomly crop the image.', type=str2bool, default=False)
parser.add_argument('--crop_height', help='The height to crop the image.', type=int, default=256)
parser.add_argument('--crop_width', help='The width to crop the image.', type=int, default=256)
parser.add_argument('--batch_size', help='The training batch size.', type=int, default=5)
parser.add_argument('--valid_batch_size', help='The validation batch size.', type=int, default=1)
parser.add_argument('--num_epochs', help='The number of epochs to train for.', type=int, default=100)
parser.add_argument('--initial_epoch', help='The initial epoch of training.', type=int, default=0)
parser.add_argument('--h_flip', help='Whether to randomly flip the image horizontally.', type=str2bool, default=False)
parser.add_argument('--v_flip', help='Whether to randomly flip the image vertically.', type=str2bool, default=False)
parser.add_argument('--brightness', help='Randomly change the brightness (list).', type=float, default=None, nargs='+')
parser.add_argument('--rotation', help='The angle to randomly rotate the image.', type=float, default=0.)
parser.add_argument('--zoom_range', help='The times for zooming the image.', type=float, default=0., nargs='+')
parser.add_argument('--channel_shift', help='The channel shift range.', type=float, default=0.)
parser.add_argument('--data_aug_rate', help='The rate of data augmentation.', type=float, default=0.)
parser.add_argument('--checkpoint_freq', help='How often to save a checkpoint.', type=int, default=1)
parser.add_argument('--validation_freq', help='How often to perform validation.', type=int, default=1)
parser.add_argument('--num_valid_images', help='The number of images used for validation.', type=int, default=20)
parser.add_argument('--data_shuffle', help='Whether to shuffle the data.', type=str2bool, default=True)
parser.add_argument('--random_seed', help='The random shuffle seed.',type=int, default=None)
parser.add_argument('--weights', help='The path of weights to be loaded.', type=str, default=None)
parser.add_argument('--steps_per_epoch', help='The training steps of each epoch', type=int, default=None)
parser.add_argument('--lr_scheduler', help='The strategy to schedule learning rate.', type=str, default='cosine_decay',
                    choices=['expo_decay', 'poly_decay', 'cosine_decay'])
parser.add_argument('--lr_warmup', help='Whether to use lr warm up.', type=bool, default=False)
parser.add_argument('--learning_rate', help='The initial learning rate.', type=float, default=3e-4)
parser.add_argument('--optimizer', help='The optimizer for training.', type=str, default='adam',
                    choices=['sgd', 'adam', 'nadam', 'adamw', 'nadamw', 'sgdw'])

args = parser.parse_args()

# check related paths
paths = check_related_path(os.getcwd())

# get image and label file names for training and validation
train_image_names, valid_image_names = get_dataset_info(args.dataset)

# quantization model
quantize_model = tfmot.quantization.keras.quantize_model

# build the model
net, base_model = builder(args.num_classes, (args.crop_height, args.crop_width), args.model, args.base_model)

is_quantize = True

if is_quantize:
    net = quantize_model(net)

# summary
# net.summary()

# load weights
if args.weights is not None:
    print('Loading the weights...')
    net.load_weights(args.weights)

# chose loss
losses = {'ce': categorical_crossentropy_with_logits,
          'focal_loss': focal_loss(alpha=0.25,gamma=4.0),
          'miou_loss': miou_loss(num_classes=args.num_classes),
          'self_balanced_focal_loss': self_balanced_focal_loss(alpha=1,gamma=2.0),
          'dice_loss': dice_coef_loss()}

loss = losses[args.loss] if args.loss is not None else categorical_crossentropy_with_logits

# chose optimizer
total_iterations = len(train_image_names) * args.num_epochs // args.batch_size

# lr schedule strategy
if args.lr_warmup and args.num_epochs - 5 <= 0:
    raise ValueError('num_epochs must be larger than 5 if lr warm up is used.')

lr_decays = {'poly_decay':PolynomialDecay(initial_learning_rate=args.learning_rate,decay_steps=len(train_image_names),end_learning_rate=1e-10,name='poly_lr'),
             'cosine_decay':CosineDecay(initial_learning_rate=args.learning_rate, decay_steps=len(train_image_names),name='cosine_lr'),
             'expo_decay':ExponentialDecay(initial_learning_rate=args.learning_rate, decay_steps=len(train_image_names),decay_rate=0.96,name='expo_lr')}

lr_decay = lr_decays[args.lr_scheduler]

optimizers = {'adam': tf.keras.optimizers.Adam(learning_rate=lr_decay),
              'sgd': tf.keras.optimizers.SGD(learning_rate=lr_decay, momentum=0.99)
              }

# training and validation steps
steps_per_epoch = len(train_image_names) // args.batch_size if not args.steps_per_epoch else args.steps_per_epoch
validation_steps = len(valid_image_names) // args.valid_batch_size

# data augmentation setting
image_gen = ImageDataGenerator(random_crop=args.random_crop,
                               rotation_range=args.rotation,
                               brightness_range=args.brightness,
                               zoom_range=args.zoom_range,
                               channel_shift_range=args.channel_shift,
                               horizontal_flip=args.v_flip,
                               vertical_flip=args.v_flip)

train_generator = image_gen.flow(images_list=train_image_names,
                                 num_classes=args.num_classes,
                                 batch_size=args.batch_size,
                                 target_size=(args.crop_height, args.crop_width),
                                 shuffle=args.data_shuffle,
                                 seed=args.random_seed,
                                 data_aug_rate=args.data_aug_rate)

valid_generator = image_gen.flow(images_list=valid_image_names,
                                 num_classes=args.num_classes,
                                 batch_size=args.valid_batch_size,
                                 target_size=(args.crop_height, args.crop_width),
                                 data_aug_rate=args.data_aug_rate
                                 )

# begin training
print("\n***** Begin training *****")
print("Dataset -->", args.dataset)
print("Num Images -->", len(train_image_names))
print("Model -->", args.model)
print("Base Model -->", base_model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Epochs -->", args.num_epochs)
print("Initial Epoch -->", args.initial_epoch)
print("Batch Size -->", args.batch_size)
print("Num Classes -->", args.num_classes)

print("Data Augmentation:")
print("\tData Augmentation Rate -->", args.data_aug_rate)
print("\tVertical Flip -->", args.v_flip)
print("\tHorizontal Flip -->", args.h_flip)
print("\tBrightness Alteration -->", args.brightness)
print("\tRotation -->", args.rotation)
print("\tZoom -->", args.zoom_range)
print("\tChannel Shift -->", args.channel_shift)

print("")

epochs = args.num_epochs
global_steps = 0
optimizer = optimizers[args.optimizer]
net.compiled_metrics = None
ce_loss = categorical_crossentropy_with_logits

for epoch in range(epochs):
    epoch_loss = 0
    batches = 0
    with tqdm(total=len(train_image_names),desc=f'Epoch {epoch+1}/{epochs}',unit='img') as pbar:
        for i,(images, masks) in enumerate(train_generator):
            with tf.GradientTape() as tape:
                preds = net(images,training=True)
                true_masks = tf.cast(masks,tf.float32)
                loss_val = loss(true_masks,preds)

            # update model
            grad = tape.gradient(loss_val,net.trainable_variables)
            optimizer.apply_gradients(zip(grad,net.trainable_variables))

            epoch_loss += loss_val.numpy()

            pbar.update(images.shape[0])
            global_steps += 1
            pbar.set_postfix(**{'loss(batch)':loss_val.numpy()})

            # Evaluation
            # 특정 시점마다 validation을 진행하도록 해야 한다. 한 epoch에서 몇 번 진행할지 설정해야함.

            division_step = (len(train_image_names) // (2 * args.batch_size))
            export_img_step = (len(train_image_names) // (4 * args.batch_size))

            if division_step > 0 or export_img_step > 0:
        
                if global_steps % division_step == 0:
                    val_score = evaluate(net,valid_generator)

            # 명시적으로 loop를 나가게 해줘야함. generator loop가 무한으로 돌기때문에
            batches += 1
            if batches >= (len(train_image_names) / args.batch_size):
                break

    # save weights
    net.save(filepath=os.path.join(
        paths['weights_path'], '{model}_based_on_{base_model}_{epoch}.h5'.format(model=args.model, base_model=base_model,epoch=epoch)))




