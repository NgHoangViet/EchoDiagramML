import torch
import torch.nn as nn
import torchmetrics
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# Hyper-parameters
num_classes = 3
batch_size = 100
learning_rate = 0.001

# Data dir
train_dir = 'DATA_CHAMBER_2021/train'
test_dir = 'DATA_CHAMBER_2021/test'

# Data transforms
training_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
testing_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])


# Need Resize(32,32)
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
        # W - F + 1
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


class Resnet18(pl.LightningModule):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.features = models.resnet18(pretrained=True)
        # Freeze all layers
        for param in self.features.parameters():
            param.requires_grad = False
        # change the last layer
        num_ftrs = self.features.fc.in_features
        self.features.fc = torch.nn.Linear(num_ftrs, 3)

        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        out = self.features(x)

        return out

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
        return torch.optim.Adam(self.parameters(), lr=learning_rate)


class GoogleNet(pl.LightningModule):
    def __init__(self):
        super(GoogleNet, self).__init__()
        self.features = models.googlenet(pretrained=True)
        # Freeze all layers
        for param in self.features.parameters():
            param.requires_grad = False
        # change the last layer
        num_ftrs = self.features.fc.in_features
        print(num_ftrs)
        self.features.fc = torch.nn.Linear(num_ftrs, 3)

        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        out = self.features(x)

        return out

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
        return torch.optim.Adam(self.parameters(), lr=learning_rate)


class VGG16(pl.LightningModule):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = models.vgg16(pretrained=True)
        # Freeze all layers
        for param in self.features.parameters():
            param.requires_grad = False
        # change the last layer
        self.features.classifier[6] = torch.nn.Linear(4096, 3, bias=True)
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        out = self.features(x)

        return out

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
        return torch.optim.Adam(self.parameters(), lr=learning_rate)


if __name__ == '__main__':
    model = VGG16()

    trainer = Trainer(auto_lr_find=True, max_epochs=5, fast_dev_run=False, auto_scale_batch_size=True)
    trainer.fit(model)
    trainer.test(model)

    # fast_dev_run=True -> runs single batch through training and validation
    # auto_lr_find: automatically finds a good learning rate before training
    # deterministic: makes training reproducable
    # gradient_clip_val: 0 default
