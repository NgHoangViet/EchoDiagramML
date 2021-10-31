# 0) Prepare data
# 1) Design model(input size, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#   - forward pass: compute prediction
#   - backward pass: gradients
#   - update weights


import torch
import torch.nn as nn
import torchmetrics
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# Hyper-parameters
num_classes = 3
VGG16_epochs = 2
ConvNet_epochs = 5
batch_size = 100
learning_rate = 0.001

# Data dir
train_dir = 'DATA_CHAMBER_2021/train'
test_dir = 'DATA_CHAMBER_2021/test'

# Data transforms
training_transforms = transforms.Compose([transforms.Resize((32, 32)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
testing_transforms = transforms.Compose([transforms.Resize((32, 32)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


class VGG16(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.features = self._make_layers()
        self.classfier_head = nn.Linear(512, 3)
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        out = self.features(x)
        out = self.classfier_head(out.view(out.size(0), -1))
        return out

    def _make_layers(self):
        config = [64, 64, 'MP', 128, 128, 'MP', 256, 256, 256, 'MP', 512, 512, 512, 'MP', 512, 512, 512, 'MP']
        layers = []
        c_in = 3
        for c in config:
            if c == 'MP':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels=c_in, out_channels=c, kernel_size=3, padding=1),
                    nn.BatchNorm2d(c),
                    nn.ReLU6(inplace=True)
                ]
                c_in = c
        return nn.Sequential(*layers)

    def training_step(self, batch, batch_idx):
        images, labels = batch

        # Forward pass
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        self.log('train_acc_step', self.accuracy(outputs, labels))
        self.log("train_loss", loss)
        return loss

    # define what happens for testing here

    def train_dataloader(self):
        train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=training_transforms)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=4,
                                                   shuffle=True)
        return train_loader

    def test_dataloader(self):
        test_dataset = torchvision.datasets.ImageFolder(test_dir, transform=testing_transforms)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=4,
                                                  shuffle=False)
        return test_loader

    def test_step(self, batch, batch_idx):
        images, labels = batch

        # Forward pass
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        self.log('train_acc_step', self.accuracy(outputs, labels))
        return {"test_loss": loss}

    def configure_optimizers(self):
        return torch.optim.SGD(model.parameters(), lr=learning_rate)


# Fully connected neural network with one hidden layer
class ConvNet(pl.LightningModule):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 70)
        self.fc3 = nn.Linear(70, 3)  # 3 classes
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 16 * 5 * 5)  # -> n, 400
        x = F.relu(self.fc1(x))  # -> n, 120
        x = F.relu(self.fc2(x))  # -> n, 70
        x = self.fc3(x)  # -> n, 3
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch

        # Forward pass
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        self.log('train_acc', self.accuracy(outputs, labels))
        self.log("train_loss", loss)
        return loss

    def train_dataloader(self):
        train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=training_transforms)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=4,
                                                   shuffle=True)
        return train_loader

    def test_dataloader(self):
        test_dataset = torchvision.datasets.ImageFolder(test_dir, transform=testing_transforms)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=4,
                                                  shuffle=False)
        return test_loader

    def test_step(self, batch, batch_idx):
        images, labels = batch

        # Forward pass
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)

        self.log('test_acc', self.accuracy(outputs, labels))
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(model.parameters(), lr=learning_rate)


if __name__ == '__main__':
    model = ConvNet()

    trainer = Trainer(auto_lr_find=True, max_epochs=ConvNet_epochs, fast_dev_run=False, auto_scale_batch_size=True)
    trainer.fit(model)
    trainer.test(model)

    # fast_dev_run=True -> runs single batch through training and validation
    # auto_lr_find: automatically finds a good learning rate before training
    # deterministic: makes training reproducable
    # gradient_clip_val: 0 default
