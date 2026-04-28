import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from ..modules import BasicConv2d


block_map = {'BasicBlock': BasicBlock,
             'Bottleneck': Bottleneck}


class ResNet9(ResNet):
    def __init__(self, 
                 block, 
                 channels=[32, 64, 128, 256], 
                 in_channel=1, 
                 layers=[1, 2, 2, 1], 
                 strides=[1, 2, 2, 1], 
                 maxpool=True):
        if block in block_map.keys():
            block = block_map[block]
        else:
            raise ValueError(
                "Error type for -block-Cfg-, supported: 'BasicBlock' or 'Bottleneck'.")
        self.maxpool_flag = maxpool
        super(ResNet9, self).__init__(block, layers)

        # Not used #
        self.fc = None
        ############
        self.inplanes = channels[0]
        self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.conv1 = BasicConv2d(in_channel, self.inplanes, 3, 1, 1)

        self.layer1 = self._make_layer(
            block, channels[0], layers[0], stride=strides[0], dilate=False)

        self.layer2 = self._make_layer(
            block, channels[1], layers[1], stride=strides[1], dilate=False)
        self.layer3 = self._make_layer(
            block, channels[2], layers[2], stride=strides[2], dilate=False)
        self.layer4 = self._make_layer(
            block, channels[3], layers[3], stride=strides[3], dilate=False)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        if blocks >= 1:
            layer = super()._make_layer(block, planes, blocks, stride=stride, dilate=dilate)
        else:
            def layer(x): return x
        return layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool_flag:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x.shape = [batch_size, 512, 16, 16]
        return x

class ResNet50(ResNet):
    def __init__(
        self,
        block,
        channels=[64, 128, 256, 512],
        in_channel=3,  # Typically 3 for RGB images
        layers=[3, 4, 6, 3],
        strides=[1, 2, 2, 1],  # Adjusted strides to achieve (C, 16, 16) output
        maxpool=True
    ):
        """
        Initializes the ResNet50 model tailored for input shape (3, 64, 64) and output shape (C, 16, 16).

        Args:
            block (str): Type of block to use ('BasicBlock' or 'Bottleneck').
            channels (list): Number of channels for each layer.
            in_channel (int): Number of input channels.
            layers (list): Number of blocks in each layer.
            strides (list): Strides for each layer to control downsampling.
            maxpool (bool): Whether to include a max pooling layer after the initial convolution.
        """
        if block in block_map:
            block = block_map[block]
        else:
            raise ValueError(
                "Invalid block type. Supported types: 'BasicBlock' or 'Bottleneck'."
            )
        
        self.maxpool_flag = maxpool
        super(ResNet50, self).__init__(block, layers)
        
        # Initialize parameters specific to ResNet50
        self.inplanes = channels[0]
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        
        # Adjusted initial convolution: 3x3 kernel, stride=1, padding=1
        self.conv1 = BasicConv2d(in_channel, self.inplanes, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Adjusted max pooling: kernel size=3, stride=2, padding=1
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if self.maxpool_flag else nn.Identity()
        
        # Override layers with adjusted strides to control downsampling
        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=strides[0])
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=strides[1])
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=strides[2])
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=strides[3])
        
        # Typically, ResNet50 includes an average pooling and fully connected layer
        # Uncomment and modify if needed
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(channels[3] * block.expansion, num_classes)
        self.fc = None

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """
        Creates a layer consisting of multiple blocks.

        Args:
            block (class): Block class to use (e.g., Bottleneck).
            planes (int): Number of output channels for the blocks.
            blocks (int): Number of blocks to stack.
            stride (int): Stride for the first block in the layer.
            dilate (bool): Whether to apply dilation.

        Returns:
            nn.Sequential: A sequential container of blocks.
        """
        if blocks >= 1:
            layer = super()._make_layer(block, planes, blocks, stride=stride, dilate=dilate)
        else:
            def layer(x): return x
        return layer

    def forward(self, x):
        """
        Defines the forward pass of the ResNet50 model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 64, 64).

        Returns:
            torch.Tensor: Output features from the last convolutional layer of shape (C, 16, 16).
        """
        x = self.conv1(x)    # (batch_size, 256, 64, 64)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool_flag:
            x = self.maxpool(x)  # (batch_size, 256, 32, 32)

        x = self.layer1(x)  # (batch_size, 256, 32, 32)
        x = self.layer2(x)  # (batch_size, 512, 16, 16)
        x = self.layer3(x)  # (batch_size, 1024, 16, 16)
        x = self.layer4(x)  # (batch_size, 2048, 16, 16)

        # If you uncomment the avgpool and fc layers, include them here
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

class ResNet101(ResNet):
    def __init__(
        self,
        block,
        channels=[64, 128, 256, 512],
        in_channel=3,  # Typically 3 for RGB images
        layers=[3, 4, 23, 3],
        strides=[1, 2, 2, 1],  # Adjusted strides to achieve (C, 16, 16) output
        maxpool=True
    ):
        """
        Initializes the ResNet101 model tailored for input shape (3, 64, 64) and output shape (C, 16, 16).

        Args:
            block (str): Type of block to use ('BasicBlock' or 'Bottleneck').
            channels (list): Number of channels for each layer.
            in_channel (int): Number of input channels.
            layers (list): Number of blocks in each layer. Default corresponds to ResNet101.
            strides (list): Strides for each layer to control downsampling.
            maxpool (bool): Whether to include a max pooling layer after the initial convolution.
        """
        if block in block_map:
            block = block_map[block]
        else:
            raise ValueError(
                "Invalid block type. Supported types: 'BasicBlock' or 'Bottleneck'."
            )
        
        self.maxpool_flag = maxpool
        super(ResNet101, self).__init__(block, layers)
        
        # Initialize parameters specific to ResNet101
        self.inplanes = channels[0]
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        
        # Adjusted initial convolution: 3x3 kernel, stride=1, padding=1
        self.conv1 = BasicConv2d(in_channel, self.inplanes, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Adjusted max pooling: kernel size=3, stride=2, padding=1
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if self.maxpool_flag else nn.Identity()
        
        # Override layers with adjusted strides to control downsampling
        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=strides[0])
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=strides[1])
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=strides[2])
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=strides[3])
        
        # Typically, ResNet101 includes an average pooling and fully connected layer
        # Uncomment and modify if needed
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(channels[3] * block.expansion, num_classes)
        self.fc = None

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """
        Creates a layer consisting of multiple blocks.

        Args:
            block (class): Block class to use (e.g., Bottleneck).
            planes (int): Number of output channels for the blocks.
            blocks (int): Number of blocks to stack.
            stride (int): Stride for the first block in the layer.
            dilate (bool): Whether to apply dilation.

        Returns:
            nn.Sequential: A sequential container of blocks.
        """
        if blocks >= 1:
            layer = super()._make_layer(block, planes, blocks, stride=stride, dilate=dilate)
        else:
            def layer(x): return x
        return layer

    def forward(self, x):
        """
        Defines the forward pass of the ResNet101 model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 64, 64).

        Returns:
            torch.Tensor: Output features from the last convolutional layer of shape (C, 16, 16).
        """
        x = self.conv1(x)    # (batch_size, 256, 64, 64)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool_flag:
            x = self.maxpool(x)  # (batch_size, 256, 32, 32)

        x = self.layer1(x)  # (batch_size, 256, 32, 32)
        x = self.layer2(x)  # (batch_size, 512, 16, 16)
        x = self.layer3(x)  # (batch_size, 1024, 16, 16)
        x = self.layer4(x)  # (batch_size, 2048, 16, 16)

        # If you uncomment the avgpool and fc layers, include them here
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x
