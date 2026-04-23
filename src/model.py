"""Multimodal 3D ResNet-18 model for 5-class brain MRI classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    """A basic 3D residual block with optional projection shortcut."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        """Initialize the residual block.

        Args:
            in_ch: Number of input channels.
            out_ch: Number of output channels.
            stride: Stride for the first convolution in the main path.
        """
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_ch)

        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block to the input tensor.

        Args:
            x: Input tensor of shape (batch, channels, D, H, W).

        Returns:
            Output tensor after residual addition and ReLU.
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class BranchResNet18(nn.Module):
    """A single-modality 3D ResNet-18 branch that outputs a 512D feature vector."""

    def __init__(self) -> None:
        """Initialize one 3D ResNet-18 branch for a single MRI modality."""
        super().__init__()

        self.stem_conv = nn.Conv3d(
            in_channels=1,
            out_channels=32,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.stem_bn = nn.BatchNorm3d(32)
        self.stem_relu = nn.ReLU(inplace=True)
        self.stem_pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(in_ch=32, out_ch=32, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(in_ch=32, out_ch=64, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(in_ch=64, out_ch=128, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(in_ch=128, out_ch=256, num_blocks=2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()

    def _make_layer(self, in_ch: int, out_ch: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Create a ResNet stage composed of residual blocks.

        Args:
            in_ch: Number of input channels to the first block.
            out_ch: Number of output channels for blocks in this stage.
            num_blocks: Number of residual blocks in the stage.
            stride: Stride for the first block.

        Returns:
            A sequential stage of residual blocks.
        """
        blocks = [ResidualBlock3D(in_ch=in_ch, out_ch=out_ch, stride=stride)]
        for _ in range(1, num_blocks):
            blocks.append(ResidualBlock3D(in_ch=out_ch, out_ch=out_ch, stride=1))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single-modality branch.

        Args:
            x: Input tensor of shape (batch, 1, 128, 128, 128).

        Returns:
            Feature tensor of shape (batch, 256).
        """
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_relu(x)
        x = self.stem_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = self.flatten(x)
        return x


class MultiModalBrainClassifier(nn.Module):
    """Four-branch multimodal 3D ResNet-18 classifier for brain MRI."""

    def __init__(self, num_classes: int = 5) -> None:
        """Initialize the multimodal classifier.

        Args:
            num_classes: Number of output classes.
        """
        super().__init__()

        self.branch_t1 = BranchResNet18()
        self.branch_t1ce = BranchResNet18()
        self.branch_t2 = BranchResNet18()
        self.branch_flair = BranchResNet18()

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GroupNorm(1, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.GroupNorm(1, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for multimodal MRI classification.

        Args:
            x: Input tensor of shape (batch, 4, 128, 128, 128).

        Returns:
            Logits tensor of shape (batch, 5).
        """
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input tensor, got shape {tuple(x.shape)}")
        if x.size(1) != 4:
            raise ValueError(f"Expected 4 channels (T1, T1CE, T2, FLAIR), got {x.size(1)}")

        x_t1 = x[:, 0:1, :, :, :]
        x_t1ce = x[:, 1:2, :, :, :]
        x_t2 = x[:, 2:3, :, :, :]
        x_flair = x[:, 3:4, :, :, :]

        f_t1 = self.branch_t1(x_t1)
        f_t1ce = self.branch_t1ce(x_t1ce)
        f_t2 = self.branch_t2(x_t2)
        f_flair = self.branch_flair(x_flair)

        fused = torch.cat([f_t1, f_t1ce, f_t2, f_flair], dim=1)
        logits = self.classifier(fused)
        return logits


def get_model(device: str = "cuda") -> MultiModalBrainClassifier:
    """Create and move the multimodal classifier to the target device.

    Args:
        device: Target device string, for example "cuda" or "cpu".

    Returns:
        A device-mapped MultiModalBrainClassifier instance.
    """
    model = MultiModalBrainClassifier(num_classes=5)
    model = model.to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    """Print parameter statistics for the multimodal model.

    Args:
        model: The model to inspect.

    Returns:
        Total parameter count.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    branch_t1_params = sum(p.numel() for p in model.branch_t1.parameters())
    branch_t1ce_params = sum(p.numel() for p in model.branch_t1ce.parameters())
    branch_t2_params = sum(p.numel() for p in model.branch_t2.parameters())
    branch_flair_params = sum(p.numel() for p in model.branch_flair.parameters())
    fusion_params = sum(p.numel() for p in model.classifier.parameters())

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Branch T1 parameters: {branch_t1_params:,}")
    print(f"Branch T1CE parameters: {branch_t1ce_params:,}")
    print(f"Branch T2 parameters: {branch_t2_params:,}")
    print(f"Branch FLAIR parameters: {branch_flair_params:,}")
    print(f"Fusion layer parameters: {fusion_params:,}")

    return total_params


if __name__ == "__main__":
    """Run a quick model sanity check."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = get_model(device=device)
    x = torch.randn(2, 4, 128, 128, 128, device=device)
    out = model(x)

    print(f"Input shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(out.shape)}")

    count_parameters(model)

    assert out.shape == (2, 5), f"Expected output shape (2, 5), got {tuple(out.shape)}"
    print("Model sanity check passed")
