import torch
import torch.nn as nn
import torchvision.models as models


class ResNetBackbone(nn.Module):
    """
    ResNet-18 backbone for EXP_08 CNN-LSTM-MDN Sequential Landmark Detection.
    
    Extracts layer3 features at stride 16:
        Input:  (B, 3, 512, 512)
        Output: (B, 256, 32, 32)
    
    Single-scale feature extraction — no FPN in this initial version.
    The 32x32 feature map provides ~16px spatial resolution, giving each grid cell
    a receptive field of approximately 128x128 pixels in the original image.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_dim = config.backbone_feature_dim  # 256 for ResNet-18 layer3
        
        # Load pretrained ResNet-18
        if config.backbone_pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            resnet = models.resnet18(weights=weights)
            print(f"✅ Loaded pretrained ResNet-18 (ImageNet1K_V1)")
        else:
            resnet = models.resnet18(weights=None)
            print(f"⚠️ Initialized ResNet-18 from scratch (no pretrained weights)")
        
        # Extract layers up to layer3 (stride 16)
        # layer1: (B, 64, 128, 128)   stride 4
        # layer2: (B, 128, 64, 64)    stride 8
        # layer3: (B, 256, 32, 32)    stride 16
        self.stem = nn.Sequential(
            resnet.conv1,    # (B, 64, 256, 256) stride 2
            resnet.bn1,
            resnet.relu,
            resnet.maxpool   # (B, 64, 128, 128) stride 4
        )
        self.layer1 = resnet.layer1  # (B, 64, 128, 128)
        self.layer2 = resnet.layer2  # (B, 128, 64, 64)
        self.layer3 = resnet.layer3  # (B, 256, 32, 32)
        
        # Projection to match LSTM hidden dim if backbone_feature_dim != lstm_hidden_dim
        if config.backbone_feature_dim != config.lstm_hidden_dim:
            self.proj = nn.Conv2d(config.backbone_feature_dim, config.lstm_hidden_dim, kernel_size=1)
        else:
            self.proj = nn.Identity()
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, 512, 512) normalized RGB images
            
        Returns:
            feature_map: (B, D, 32, 32) where D = lstm_hidden_dim (256)
        """
        x = self.stem(pixel_values)    # (B, 64, 128, 128)
        x = self.layer1(x)             # (B, 64, 128, 128)
        x = self.layer2(x)             # (B, 128, 64, 64)
        x = self.layer3(x)             # (B, 256, 32, 32)
        x = self.proj(x)              # (B, D, 32, 32)
        return x
    
    def get_param_groups(self, backbone_lr_mult: float = 0.1):
        """
        Returns parameter groups with differential learning rates.
        Backbone params train at backbone_lr_mult × base_lr to prevent pretrained weight drift.
        """
        backbone_params = list(self.stem.parameters()) + \
                          list(self.layer1.parameters()) + \
                          list(self.layer2.parameters()) + \
                          list(self.layer3.parameters())
        proj_params = list(self.proj.parameters()) if not isinstance(self.proj, nn.Identity) else []
        
        return [
            {"params": backbone_params, "lr_mult": backbone_lr_mult},
            {"params": proj_params, "lr_mult": 1.0}
        ]
