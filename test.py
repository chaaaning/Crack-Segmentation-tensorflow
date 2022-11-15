from utils.data_generator import ImageDataGenerator
from utils.helpers import get_dataset_info, check_related_path
from utils.losses import categorical_crossentropy_with_logits
from utils.metrics import MeanIoU
from builders import builder
import tensorflow as tf
import argparse
import os
from eval import evaluate
import tensorflow_model_optimization as tfmot


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
parser.add_argument('--dataset', help='The path of the dataset.', type=str, required=True)
parser.add_argument('--num_classes', help='The number of classes to be segmented.', type=int, required=True)
parser.add_argument('--crop_height', help='The height to crop the image.', type=int, default=256)
parser.add_argument('--crop_width', help='The width to crop the image.', type=int, default=256)
parser.add_argument('--batch_size', help='The training batch size.', type=int, default=5)
parser.add_argument('--weights', help='The path of weights to be loaded.', type=str, default=None)

args = parser.parse_args()

# check related paths
paths = check_related_path(os.getcwd())

# image list loads
test_image_names = os.listdir('../data/test')
test_image_names = [os.path.join(args.dataset,name) for name in test_image_names]

# quantization model
quantize_model = tfmot.quantization.keras.quantize_model

# build the model
net, base_model = builder(args.num_classes, (args.crop_height, args.crop_width), args.model, args.base_model)
net = quantize_model(net)

# summary
net.summary()

# load weights
print('Loading the weights...')
net.load_weights(filepath='./weights/UNet_based_on_MobileNetV2_CE_QAT_288384_50000.h5')

# data generator
test_gen = ImageDataGenerator()

test_generator = test_gen.flow(images_list=test_image_names,
                               num_classes=args.num_classes,
                               batch_size=args.batch_size,
                               target_size=(args.crop_height, args.crop_width)
                               )

# begin testing
print("\n***** Begin testing *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Base Model -->", base_model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Batch Size -->", args.batch_size)
print("Num Classes -->", args.num_classes)

print("")

# testing
scores = evaluate(net,test_generator)

print('Dice Score={0:0.4f}'.format(scores))
