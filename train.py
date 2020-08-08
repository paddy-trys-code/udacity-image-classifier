import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

def arg_parser():
    parser = argparse.ArgumentParser(description="NN Config.")
    parser.add_argument('--arch', type=str, help='name desired model')
    parser.add_argument('--save_dir', type=str, help='name checkpoint folder')
    parser.add_argument('--learning_rate', type=float, help='specify learning rate (float)')
    parser.add_argument('--hidden_units', type=int, help='specify Hidden units (int)')
    parser.add_argument('--epochs', type=int, help='specify # epochs (int)')
    parser.add_argument('--gpu', action="store_true", help='Use Cuda for parallel computation')
    args = parser.parse_args()
    return args

#modulate training data
def train_data_transform(train_dir):
    train_transforms = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    return train_data

#modulate testing data
def test_data_transform(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data

#create dataloader from dataset
def load_data(data, train=True):
    if train:
        loader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True)
    else:
        loader = torch.utils.data.DataLoader(data, batch_size=50)
    return loader

# use CUDA if available else CPU
def check_gpu(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device == "cpu":
        print("no coke, pepsi")
    return device

def model_load(architecture="vgg16"):
    if type(architecture) == type(None):
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
    else:
        exec("model = models.{}(pretrained=True)".format(architecture))
        model.name = architecture

    for param in model.parameters():
        param.requires_grad = False
    return model

def initial_classifier(model, hidden_units):
    if type(hidden_units) == type(None):
        hidden_units = 2048

    input_features = model.classifier[0].in_features

    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_features, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, 256)),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(256, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                            ]))
    return classifier


def network_trainer(model, trainloader, testloader, device,
                  criterion, optimizer, epochs, print_every, steps):
    # Check Model Kwarg
    if type(epochs) == type(None):
        epochs = 1

    print("Training \n")
    model.to("cuda")
    running_loss=0

    # Train Model
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            #Forward pass
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            # Backward pass
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')

                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    valid_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Loss: {running_loss/print_every:.3f}.. "
                  f"Validation Loss: {valid_loss/len(validloader):.3f}.. "
                  f"Accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()

    return model

#good
def validate(model, testloader, Device):
   # Do validation on the test set
    print("Testing\n")

    correct = 0
    total = 0
    model.to('cuda')
    with torch.no_grad():
        model.eval
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy achieved by the network on test images is: %d%%' % (100 * correct / total))

#good
def initial_checkpoint(model, Save_Dir, train_data):
    print("loading model")
    model.class_to_idx = train_data.class_to_idx
    #good
    checkpoint = {'input_size': 25088,
              'output_size': 102,
              'arch': 'vgg16',
              'classifier' : model.classifier,
              'learning_rate': 0.001,
              'class_to_idx': model.class_to_idx,
              'state_dict': model.state_dict(),
             }
    torch.save(checkpoint, 'save_new_checkpoint.pth')


def main():

    # Get Keyword Args for Training
    args = arg_parser()

    # Set directory for training
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Pass transforms in, then create trainloader
    train_data = train_data_transform(train_dir)
    valid_data = test_data_transform(valid_dir)
    test_data = test_data_transform(test_dir)

    trainloader = load_data(train_data)
    validloader = load_data(valid_data, train=False)
    testloader = load_data(test_data, train=False)

    # Load Model
    model = model_load(architecture=args.arch)

    # Build Classifier
    model.classifier = initial_classifier(model,
                                         hidden_units=args.hidden_units)

    # Check for GPU
    device = check_gpu(gpu_arg=args.gpu);

    # Send model to device
    model.to(device);

    # Check for learnrate args
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print("Learning rate specificed as 0.001")
    else: learning_rate = args.learning_rate

    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Define deep learning method
    print_every = 5
    steps = 0



    # Train the classifier layers using backpropogation
    trained_model = network_trainer(model, trainloader, validloader,
                                  device, criterion, optimizer, args.epochs,
                                  print_every, steps)

    print("\ntraining done")

    # Quickly Validate the model
    validate(trained_model, testloader, device)

    # Save the model
    initial_checkpoint(trained_model, args.save_dir, train_data)


#run
if __name__ == '__main__': main()
