'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    This is the basic building block of ResNet.
    It consists of two convolutional layers with batch normalization and a residual connection.

    The idea is that the input can skip over the two convolutions via a shortcut connection.
    This helps prevent vanishing gradients and makes training deeper networks easier.
    """
    expansion = 1  # No channel expansion in the basic block

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # If the input and output dimensions don't match (because of stride > 1), we can use a 1x1 convolutional layer
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # First convolutional layer with activation
        out = self.bn2(self.conv2(out))  # Second convolutional layer
        out += self.shortcut(x)  # Add the shortcut
        return F.relu(out)  # Final activation


class ResNet(nn.Module):
    """
    The ResNet model, which has been modified to -
    - Accept grayscale images (1-channel) instead of RGB (3-channel).
    - Perform binary classification (cat v dog) by setting num_classes=2.

    Apart from that, the model follows the standard ResNet design -
    - A convolutional layer at the start
    - Four groups of residual blocks
    - A global average pooling layer
    - A fully connected layer for classification
    """

    def __init__(self, block, num_blocks, num_classes=2):  # num_classes=2 for binary classification
        super(ResNet, self).__init__()
        self.in_planes = 64  # This should help keep track of the current number of channels

        # The first modified convolution layer to take grayscale images
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # These are the layers of the ResNet
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # The final linear layer modified for binary classification
        self.linear = nn.Linear(2048, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        Creates a sequence of residual blocks.
        The first block may need downsampling (stride > 1), while the others keep stride=1.
        """
        strides = [stride] + [1] * (num_blocks - 1)  # The first block in each layer might have a stride of 2
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion  # Update in_planes for the next layer
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        This should define the forward pass through the network

        - Applying an initial convolution and batch normalization
        - Passing through four residual block layers
        - Using global average pooling to reduce the feature map to a vector
        - Pass through a fully connected layer for classification.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)  # Global average pooling which reduces spatial dimensions
        out = out.view(out.size(0), -1)  # Flatten into a vector
        return self.linear(out)  # Pass through the final classification layer


# Function to create a ResNet-18 model
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])  # Standard ResNet-18 architecture


def test():
    """
    Quick test to check if the model works with grayscale images.
    """
    net = ResNet18()
    y = net(torch.randn(1, 1, 64, 64))  # 1 sample, 1-channel image of size 64x64
    print(y.size())  # Should output: torch.Size([1, 2])

if __name__ == "__main__":
    test()  # This runs a quick test
