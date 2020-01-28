# necessary imports here

import sys
import torch
import argparse
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# architecture nodes
arch_nodes = {
    "alexnet": 9216,
    "vgg16": 25088,
    "densenet121": 1024,
}


class Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers,
                 drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.'''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(input_size, hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend(
            [nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''

        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)

        return F.log_softmax(x, dim=1)


# create a helper function to initiate a classifier
def model_init(arch='vgg16', output_size=102,
               hidden_layers=[512, 256],
               dropout=0.5, device='cuda', lr=0.001):
    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        print(
            "Current implementation only supports alexnet, vgg16 or densenet121. Try again.")
        sys.exit()

    # freeze the parameters so that we only backpropagate through our own classifier
    for param in model.parameters():
        param.requires_grad = False

    classifier = Classifier(arch_nodes[arch], output_size,
                            hidden_layers, dropout)
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    model.to(device)

    return model, optimizer, criterion


# function that trains the network
def train(model, trainloader, validationloader, criterion, optimizer,
          epochs=5, print_every=40):
    steps = 0
    running_loss = 0

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            # print(inputs.shape)
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                loss_ = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validationloader:
                        inputs, labels = inputs.to(device), labels.to(
                            device)
                        logps = model.forward(inputs)
                        loss_ += criterion(logps, labels)
                        validation_loss += loss_.item()

                        # calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(
                            *top_class.shape)
                        accuracy += torch.mean(
                            equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{epochs} - Step {steps}")
                print(
                    f"Train Loss:{running_loss / print_every:0.3f} ")
                print(
                    f"Validation Loss:{validation_loss / len(validationloader):0.3f} ")
                print(
                    f"Validation Accuracy:{accuracy / len(validationloader):0.3f} ")
                print("=" * 50)
                running_loss = 0
                model.train()
    print("All Epochs Completed")


# function that validates network on test data
def validation(model, testloader):
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)

            # calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(
                equals.type(torch.FloatTensor)).item()
    accuracy_score = accuracy / len(testloader)
    print(f"Accuracy on test set: {accuracy_score:0.2f}")


# function that saves the model
def save_checkpoint(model, optimizer, save_directory=""):
    print(f"Attempting to save trained model in {save_dir}")
    try:
        model.class_to_idx = train_data.class_to_idx
        checkpoint = {
            'arch': 'vgg16',
            'output_size': 102,
            'hidden_layers': [512, 256],
            'state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx,
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, save_directory + '/checkpoint.pth')
        # print("Saved model with following parameters::")
        # print(checkpoint)
    except Exception as msg:
        print("Error while saving checkpoint")
        print(str(msg))


if __name__ == '__main__':
    # construct the argument parser
    ag = argparse.ArgumentParser()
    ag.add_argument("--data_dir", required=False,
                    help='path to training data set')
    ag.add_argument("--save_dir", required=False,
                    help='model checkpoint save path')
    ag.add_argument("--arch", required=False,
                    help='network architecture')
    ag.add_argument("--learning_rate", required=False,
                    help='training learning rate')
    ag.add_argument("--hidden_units", required=False,
                    help='list of nodes in the hidden layers')
    ag.add_argument("--epochs", required=False,
                    help='epochs for the training')
    ag.add_argument("--device", required=False,
                    help='device to run training on')

    # check the command line commands and populate with appropriate values
    args = vars(ag.parse_args())

    try:
        data_dir = str(args["data_dir"] or "flowers")
        save_dir = str(args["save_dir"] or "/home/mhasan3")
        arch = str(args["arch"] or "vgg16")
        lr = float(args["learning_rate"] or 0.001)

        hidden_raw_args = args["hidden_units"]
        if not hidden_raw_args:
            hidden_layers = [512, 256]
        else:
            try:
                hidden_layers = list(map(int,
                                         hidden_raw_args.strip('[]')
                                         .split(',')))
            except Exception as msg:
                print("Incompatible value for hidden layers. Please "
                      "supply a list of hidden layer nodes::")
                print(str((msg)))
                sys.exit()

        epochs = int(args["epochs"] or 2)
        device_str = str(args["device"] or "cpu")
        device = torch.device(device_str)
    except Exception as msg:
        print("Error in command line arguments::")
        print(str(msg))
        sys.exit()

    # # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    validation_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + '/train',
                                      transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test',
                                     transform=test_transforms)
    validation_data = datasets.ImageFolder(data_dir + '/valid',
                                           transform=validation_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data,
                                              batch_size=64,
                                              shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validationloader = torch.utils.data.DataLoader(validation_data,
                                                   batch_size=64)

    # make your own classifier
    # this is where execution begins
    print("Beginning model training based on following arguments::")
    print(f"Data Directory: {data_dir}")
    print(f"Save Directory: {save_dir}")
    print(f"Model Architecture: {arch}")
    print(f"Learning Rate: {lr}")
    print(f"Hidden Layers: {hidden_layers}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    print("=" * 50)

    # 1. Initialise a model according to given arguments
    model, optimizer, criterion = model_init(arch=arch,
                                             output_size=102,
                                             hidden_layers=hidden_layers,
                                             dropout=0.2,
                                             device=device,
                                             lr=0.001)
    print("Model Initialised. Training will begin......")
    print("=" * 50)
    # 2. Train the initialised model
    print("Model Training has started.")
    train(model=model,
          trainloader=trainloader,
          validationloader=validationloader,
          criterion=criterion,
          optimizer=optimizer,
          epochs=epochs,
          print_every=40)
    print("Model Training Ended.")
    print("=" * 50)

    # 3. Test the trained network
    print("Testing the trained network on test data")
    validation(model, testloader=testloader)
    print("Testing complete")
    print("=" * 50)

    # 4. Save the trained model
    print(
        f"Saving the trained model as a checkpoint in {save_dir} ......")
    save_checkpoint(model=model,
                    save_directory=save_dir,
                    optimizer=optimizer)
    print("Saving trained model complete.")
    print("=" * 50)
