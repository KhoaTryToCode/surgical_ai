import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure repos/TopoNet is in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../repos/TopoNet'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from depth_anything_v2.dpt import DepthAnythingV2
from models.resnet import ResNet34
from models.context_modules import get_context_module
from models.model_utils import ConvBNAct, Swish
from models.decoder import Decoder
from models.befusion import BeFusion
try:
    from DSCNet.ds_encoder import DSCNet_Encoder
except ImportError:
    DSCNet_Encoder = None


def _safe_interpolate_area(x, size):
    """Area interpolation with automatic MPS CPU fallback for non-divisible sizes."""
    if x.device.type == 'mps':
        return F.interpolate(x.cpu(), size=size, mode='area').to(x.device)
    return F.interpolate(x, size=size, mode='area')


class SimpleConcatFusion(nn.Module):
    """
    Simple concatenation baseline replacing BTF (Boundary-Aware Topological Fusion).
    Merges RGB and depth feature maps via 1x1 convolution.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb_feat, depth_feat, prev_feat=None):
        if prev_feat is not None and prev_feat.shape[2:] != rgb_feat.shape[2:]:
            prev_feat = F.interpolate(prev_feat, size=rgb_feat.shape[2:], mode='bilinear', align_corners=False)
        fused = self.conv(torch.cat([rgb_feat, depth_feat], dim=1))
        if prev_feat is not None and prev_feat.shape[1] == fused.shape[1]:
            fused = fused + prev_feat
        return fused, fused


class TopoNetAblationModel(nn.Module):
    """
    Unified TopoNet Model supporting all 6 official paper ablation modes:
      1. 'full': Full TopoNet (Snake DSCNet + BTF + clDice + Betti)
      2. 'baseline': Standard Conv + Simple Concat (No BTF, no topo losses)
      3. 'wo_lper': Snake DSCNet + BTF + Soft Dice + clDice (no Betti)
      4. 'wo_lcl': Snake DSCNet + BTF + Soft Dice + Betti (no clDice)
      5. 'wo_lper_lcl': Snake DSCNet + BTF + Soft Dice only (no topo loss)
      6. 'wo_btf': Snake DSCNet + Simple Concat + Soft Dice + clDice + Betti
    """
    def __init__(self, ablation_mode='full', depth_path=None, num_classes=4, height=1024, width=1024):
        super().__init__()
        self.ablation_mode = ablation_mode
        self.depth_path = depth_path

        # 1. Depth Anything V2 Foundation Model
        depth_configs = {
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]}
        }
        self.depth_encoder = DepthAnythingV2(**depth_configs['vitb'])
        if depth_path and os.path.exists(depth_path):
            state_dict = torch.load(depth_path, map_location='cpu')
            self.depth_encoder.load_state_dict(state_dict)
            print(f"✅ Loaded Depth Anything V2 weights from: {depth_path}")
        else:
            print(f"⚠️  Depth weights not found at '{depth_path}'. Initialized with random weights (OK for smoke tests).")
        self.depth_encoder.eval()
        self.depth_encoder.requires_grad_(False)

        # 2. Depth Feature Extractor (Snake DSCNet vs Standard Conv)
        self.use_snake = (ablation_mode != 'baseline')
        if self.use_snake:
            global DSCNet_Encoder
            if DSCNet_Encoder is None:
                from DSCNet.ds_encoder import DSCNet_Encoder
            self.dsc_encoder = DSCNet_Encoder()
        else:
            # Baseline uses standard ResNet blocks for depth
            self.dsc_encoder = ResNet34(input_channels=3, pretrained_on_imagenet=False)

        # 3. RGB Encoder (ResNet-34)
        self.rgb_encoder = ResNet34(input_channels=3, pretrained_on_imagenet=False)

        # 4. Multi-Modal Fusion (BTF vs Simple Concat)
        self.use_btf = (ablation_mode not in ['baseline', 'wo_btf'])
        if self.use_btf:
            self.be0 = BeFusion(64, 512, 512, isFirst=True)
            self.be1 = BeFusion(self.rgb_encoder.down_4_channels_out, 256, 256)
            self.be2 = BeFusion(self.rgb_encoder.down_8_channels_out, 128, 128)
            self.be3 = BeFusion(self.rgb_encoder.down_16_channels_out, 64, 64)
            self.be4 = BeFusion(self.rgb_encoder.down_32_channels_out, 32, 32, isLast=True)
        else:
            self.be0 = SimpleConcatFusion(64)
            self.be1 = SimpleConcatFusion(self.rgb_encoder.down_4_channels_out)
            self.be2 = SimpleConcatFusion(self.rgb_encoder.down_8_channels_out)
            self.be3 = SimpleConcatFusion(self.rgb_encoder.down_16_channels_out)
            self.be4 = SimpleConcatFusion(self.rgb_encoder.down_32_channels_out)

        # Skip connections
        channels_decoder = [128, 128, 128]
        self.skip_layer1 = nn.Sequential(
            ConvBNAct(self.rgb_encoder.down_4_channels_out, channels_decoder[2], kernel_size=1, activation=nn.ReLU(inplace=True))
        )
        self.skip_layer2 = nn.Sequential(
            ConvBNAct(self.rgb_encoder.down_8_channels_out, channels_decoder[1], kernel_size=1, activation=nn.ReLU(inplace=True))
        )
        self.skip_layer3 = nn.Sequential(
            ConvBNAct(self.rgb_encoder.down_16_channels_out, channels_decoder[0], kernel_size=1, activation=nn.ReLU(inplace=True))
        )

        # Context Module & Decoder
        self.context_module, channels_after_context = get_context_module(
            'ppm', self.rgb_encoder.down_32_channels_out, channels_decoder[0],
            input_size=(height // 32, width // 32), activation=nn.ReLU(inplace=True), upsampling_mode='bilinear'
        )

        self.decoder = Decoder(
            channels_in=channels_after_context, channels_decoder=channels_decoder,
            activation=nn.ReLU(inplace=True), nr_decoder_blocks=[1, 1, 1],
            encoder_decoder_fusion='add', upsampling_mode='bilinear', num_classes=num_classes
        )

    def forward(self, image):
        # 1. On-the-fly depth estimation at (1022, 1022) [multiple of ViT patch size 14]
        with torch.no_grad():
            img_depth_in = _safe_interpolate_area(image, size=(1022, 1022))
            raw_depth = self.depth_encoder.infer_image(img_depth_in)
            depth_3ch = raw_depth.expand(-1, 3, -1, -1)

        # 2. Depth Feature Extraction
        if self.use_snake:
            d0, d1, d2, d3, depth_out = self.dsc_encoder(depth_3ch)
        else:
            # Baseline standard CNN depth encoder
            out_d = self.dsc_encoder.forward_first_conv(depth_3ch)
            d0 = out_d
            out_d = F.max_pool2d(out_d, kernel_size=3, stride=2, padding=1)
            d1 = self.dsc_encoder.forward_layer1(out_d)
            d2 = self.dsc_encoder.forward_layer2(d1)
            d3 = self.dsc_encoder.forward_layer3(d2)
            depth_out = self.dsc_encoder.forward_layer4(d3)

        # 3. RGB Feature Extraction & Progressive Multi-Modal Fusion
        out_rgb = self.rgb_encoder.forward_first_conv(image)
        skipf0, out_f0 = self.be0(out_rgb, _safe_interpolate_area(d0, size=out_rgb.shape[2:]))
        out_rgb = F.max_pool2d(out_rgb, kernel_size=3, stride=2, padding=1)

        # Block 1
        out_rgb = self.rgb_encoder.forward_layer1(out_rgb)
        skipf1, out_f1 = self.be1(out_rgb, _safe_interpolate_area(d1, size=out_rgb.shape[2:]), out_f0)
        skip1 = self.skip_layer1(skipf1)

        # Block 2
        out_rgb = self.rgb_encoder.forward_layer2(out_rgb)
        skipf2, out_f2 = self.be2(out_rgb, _safe_interpolate_area(d2, size=out_rgb.shape[2:]), out_f1)
        skip2 = self.skip_layer2(skipf2)

        # Block 3
        out_rgb = self.rgb_encoder.forward_layer3(out_rgb)
        skipf3, out_f3 = self.be3(out_rgb, _safe_interpolate_area(d3, size=out_rgb.shape[2:]), out_f2)
        skip3 = self.skip_layer3(skipf3)

        # Block 4
        out_rgb = self.rgb_encoder.forward_layer4(out_rgb)
        skipf4, out_f4 = self.be4(out_rgb, _safe_interpolate_area(depth_out, size=out_rgb.shape[2:]), out_f3)

        # Context Module & Decoder
        context_out = self.context_module(out_f4)
        decoder_outs, _ = self.decoder(enc_outs=[context_out, skip3, skip2, skip1])
        logits = F.log_softmax(decoder_outs, dim=1)

        return logits, raw_depth
