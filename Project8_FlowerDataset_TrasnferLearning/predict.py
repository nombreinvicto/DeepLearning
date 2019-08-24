# necessary imports here

import sys
import torch
import argparse
import json
import pprint
from PIL import Image
from train import model_init
from torchvision import transforms

# construct the argument parser
ag = argparse.ArgumentParser()
ag.add_argument("--image", required=True,
                help="Path to image")
ag.add_argument("--checkpoint", required=True,
                help="Path to checkpoint")

ag.add_argument("--device", required=False,
                help='device to run testing on')
ag.add_argument("--topk", required=False,
                help='return how many top probabilities?')
ag.add_argument("--category_names", required=False,
                help='path to category name file')

# check the command line commands and populate with appropriate values
args = vars(ag.parse_args())

try:
    imagePath = str(args["image"])
    checkpointPath = str(args["checkpoint"])
    device_str = str(args["device"] or "cpu")
    device = torch.device(device_str)
    topk = int(args["topk"] or 5)
    category_name_fileName = str(
        args["category_names"] or "cat_to_name.json")

except Exception as msg:
    print("Error in command line arguments::")
    print(str(msg))
    sys.exit()

# print out arguments supplied
print("Beginning image classification based on following arguments::")
print(f"Image Path: {imagePath}")
print(f"Checkpoint Path: {checkpointPath}")
print(f"Top k probabilities: {topk}")
print(f"Category names Filename: {category_name_fileName}")
print(f"Device: {device}")
print("=" * 50)


# 2. function thats loads the checkpoint
def load_checkpoint(path):
    checkpoint = torch.load(path)
    model, optimizer, criterion = model_init(arch=checkpoint['arch'],
                                             output_size=checkpoint[
                                                 'output_size'],
                                             hidden_layers=checkpoint[
                                                 'hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer, criterion


# 3. function that preprocesses images supplied
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    return t(pil_image)


# 4. function that does the job of prediction
def predict(image_path, model, topk=topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    model.to(device)
    image_tensor = process_image(image_path)
    image_tensor = image_tensor.to(device)
    idx_to_class = dict(
        [[v, k] for k, v in model.class_to_idx.items()])

    with torch.no_grad():
        logps = model.forward(image_tensor.unsqueeze(0))
        ps = torch.exp(logps)
        top_p, top_ids = ps.topk(topk, dim=1)

        top_p_ = []
        for p in top_p[0]:
            p = float(p)
            top_p_.append(p)

        top_classes = []
        for ids in top_ids[0]:
            ids = float(ids)
            top_classes.append(idx_to_class[ids])

        top_class_names = []
        for class_ in top_classes:
            top_class_names.append(cat_to_name[class_])

        out_dict = {}
        for i, flower in enumerate(top_class_names):
            out_dict[flower] = {"class_no": top_classes[i],
                                "probability": top_p_[i]}

    return out_dict


# this is where execution begins
if __name__ == '__main__':
    # 0. function that loads the category names from the file
    with open(category_name_fileName, 'r') as f:
        cat_to_name = json.load(f)

    # 1. load the model
    print("Loading Checkpoint.......")
    loaded_model, loaded_optimizer, loaded_criterion = \
        load_checkpoint(checkpointPath)
    print("Checkpoint load complete.")
    print("=" * 50)

    # 2. get result from predict function
    print("Attempting to do prediction on supplied image ......")
    result_dict = predict(imagePath, loaded_model, topk=topk)
    print("Prediction complete.")
    print("=" * 50)
    print("Prediction Results::")
    pprint.pprint(result_dict)
