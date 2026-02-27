import torch
import torch.nn as nn
from torchvision.models import resnet50
import timm


class CrossAttention(nn.Module):
    def __init__(self, query_dim, kv_dim, hidden_dim):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(kv_dim, hidden_dim)
        self.value_proj = nn.Linear(kv_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key_value):
        """
        query: [B, query_dim] (ResNet features)
        key_value: [B, kv_dim] (Swin features)
        Returns:
            fused: [B, hidden_dim]
        """
        Q = self.query_proj(query).unsqueeze(1)       # [B, 1, D]
        K = self.key_proj(key_value).unsqueeze(1)     # [B, 1, D]
        V = self.value_proj(key_value).unsqueeze(1)   # [B, 1, D]

        attn_weights = torch.softmax((Q @ K.transpose(-2, -1)) * self.scale, dim=-1)  # [B, 1, 1]
        attn_output = (attn_weights @ V).squeeze(1)                                   # [B, D]
        return self.output_proj(attn_output)                                          # [B, D]


class HybridResNetSwinT_CA(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # ResNet-50 backbone
        resnet = resnet50(pretrained=True)
        self.resnet_backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC
        self.resnet_feature_dim = resnet.fc.in_features  # 2048

        # Swin-Tiny backbone (from timm)
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0, global_pool='avg')
        self.swin_feature_dim = self.swin.num_features  # 768 for Swin-Tiny

        # Cross-attention fusion (ResNet as query, Swin as key/value)
        self.fusion_dim = 512  # Tunable fusion dimension
        self.cross_attention = CrossAttention(query_dim=self.resnet_feature_dim,
                                              kv_dim=self.swin_feature_dim,
                                              hidden_dim=self.fusion_dim)

        # Final classifier
        self.classifier = nn.Linear(self.fusion_dim, num_classes)

    def forward(self, x):
        # ResNet branch
        resnet_feat = self.resnet_backbone(x)            # [B, 2048, 1, 1]
        resnet_feat = resnet_feat.view(x.size(0), -1)    # [B, 2048]

        # Swin-T branch
        swin_feat = self.swin(x)                         # [B, 768]

        # Cross-attention fusion
        fused_feat = self.cross_attention(resnet_feat, swin_feat)  # [B, fusion_dim]

        # Classification
        out = self.classifier(fused_feat)
        return out
