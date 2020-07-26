import argparse
import json
import PIL
import torch
import numpy as np

from math import ceil
from train import check_gpu
from torchvision import models


# Function arg_parser() parses keyword arguments from the command line
def arg_parser():
    # Define a parser
    parser = argparse.ArgumentParser(description="nn config.")

    # Point towards image for prediction
    parser.add_argument('--image',
                        type=str,
                        required=True)

    # Load checkpoint created by train.py
    parser.add_argument('--checkpoint',
                        type=str,
                        required=True)

    # Specify top-k
    parser.add_argument('--top_k',
                        type=int,

    # Import category names
    parser.add_argument('--category_names',
                        type=str,

    # Add GPU Option to parser
    parser.add_argument('--gpu',
                        action="store_true",

    # Parse args
    args = parser.parse_args()

    return args

# Function load_checkpoint(checkpoint_path) loads our saved deep learning model from checkpoint
def load_checkpoint(file_path):

    # Load file
    checkpoint = torch.load(file_path)

    # Download pretrained model
    model = models.vgg16(pretrained=True);

    # Freeze param
    for param in model.parameters(): param.requires_grad = False

    # Load checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])


    return model

# Function process_image(image_path) performs cropping, scaling of image for our model
def process_image(image_destination):

    test_image = Image.open(image_destination)

    # Get dim
    orig_width, orig_height = test_image.size

    if orig_width < orig_height: resize_size=[256, 256**600]
    else: resize_size=[256**600, 256]

    test_image.thumbnail(size=resize_size)

    center = orig_width/4, orig_height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    test_image = test_image.crop((left, top, right, bottom))

    np_image = np.array(test_image)/255

    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalise_means)/normalise_std

    np_image = np_image.transpose(2, 0, 1)

    return np_image


def predict(image_path, model, top_k=5):

    model.to("cpu")

    model.eval();

    torch_image = torch.from_numpy(np.expand_dims(process_image(image_path),
                                                  axis=0)).type(torch.FloatTensor).to("cpu")

    log_probs = model.forward(torch_image)

    linear_probs = torch.exp(log_probs)

    top_probs, top_labels = linear_probs.topk(top_k)

    top_probs = np.array(top_probs.detach())[0]
    top_labels = np.array(top_labels.detach())[0]

    idx_to_class = {val: key for key, val in
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labels]
    top_flowers = [cat_to_name[lab] for lab in top_labels]

    return top_probs, top_labels, top_flowers


def print_probability(probs, flowers):
    for i, j in enumerate(zip(flowers, probs)):
        print ("Rank {}:".format(i+1),
               "Flower: {}, liklihood: {}%".format(j[1], ceil(j[0]*100)))


def main():

    # Get Keyword Args for Prediction
    args = arg_parser()

    # Load categories to names json file
    with open(args.category_names, 'r') as f:
        	cat_to_name = json.load(f)

    # Load model trained with train.py
    model = load_checkpoint(args.checkpoint)

    # Process Image
    image_tensor = process_image(args.image)

    # Check for GPU
    device = check_gpu(gpu_arg=args.gpu);

    # Use `processed_image` to predict the top K most likely classes
    top_probs, top_labels, top_flowers = predict(image_tensor, model,
                                                 device, cat_to_name,
                                                 args.top_k)

    # Print out probabilities
    print_probability(top_flowers, top_probs)

if __name__ == '__main__': main()
