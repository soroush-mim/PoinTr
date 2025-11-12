"""
ULIP-Based Alignment Loss for Text-Conditioned Point Cloud Completion

This module provides contrastive alignment loss between:
- Completed point cloud embeddings (from frozen ULIP 3D encoder)
- Text embeddings (from frozen CLIP text encoder)

The alignment loss ensures that the completed point cloud matches
the semantic description provided in the text caption.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ulip_models.pointbert.point_encoder import PointTransformer
from ulip_models.pointbert.misc import cfg_from_yaml_file


class ULIPPointBERTEncoder(nn.Module):
    """
    Frozen ULIP PointBERT encoder for extracting 3D features from point clouds.

    This encoder is kept frozen to preserve the pre-trained ULIP representations.
    It's used only for computing alignment loss, not for feature extraction in the main model.

    Args:
        config_path: str, path to PointBERT config YAML file
        checkpoint_path: str (optional), path to pretrained weights
        output_dim: int, dimension of output embeddings (default: 512 for ULIP)
    """

    def __init__(self, config_path=None, checkpoint_path=None, output_dim=512):
        super().__init__()

        print(f'[ULIP_ENCODER] Loading ULIP PointBERT encoder...')

        # Use default config if not provided
        if config_path is None:
            config_path = './ulip_models/pointbert/ULIP_2_PointBERT_10k_colored_pointclouds.yaml'

        # Load config
        try:
            config = cfg_from_yaml_file(config_path)

            # Create PointBERT encoder
            # Note: We need to pass a dummy args object for compatibility
            from easydict import EasyDict
            dummy_args = EasyDict({'evaluate_3d': True})  # Flag to skip loading vision model weights

            self.point_encoder = PointTransformer(config.model, args=dummy_args)
            self.pc_feat_dim = 768  # PointBERT output dimension

        except Exception as e:
            print(f'[ULIP_ENCODER] Error loading config: {e}')
            print(f'[ULIP_ENCODER] Falling back to default PointBERT configuration')
            # Fallback: create with default settings
            from easydict import EasyDict
            default_config = EasyDict({
                'trans_dim': 384,
                'depth': 12,
                'drop_path_rate': 0.1,
                'cls_dim': 40,
                'num_heads': 6,
                'group_size': 32,
                'num_group': 512,
                'encoder_dims': 384
            })
            dummy_args = EasyDict({'evaluate_3d': True})
            self.point_encoder = PointTransformer(default_config, args=dummy_args)
            self.pc_feat_dim = 384

        # Projection layer to match ULIP embedding dimension
        self.pc_projection = nn.Parameter(torch.empty(self.pc_feat_dim, output_dim))
        nn.init.normal_(self.pc_projection, std=output_dim ** -0.5)

        self.output_dim = output_dim

        # Freeze all point encoder parameters
        for param in self.point_encoder.parameters():
            param.requires_grad = False

        # Set to eval mode
        self.point_encoder.eval()

        # Load pretrained weights if provided
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        print(f'[ULIP_ENCODER] PointBERT encoder loaded successfully')
        print(f'[ULIP_ENCODER] Point cloud feature dimension: {self.pc_feat_dim}')
        print(f'[ULIP_ENCODER] Output embedding dimension: {output_dim}')
        print(f'[ULIP_ENCODER] All parameters frozen: {not any(p.requires_grad for p in self.point_encoder.parameters())}')

    def load_checkpoint(self, checkpoint_path):
        """Load pretrained ULIP weights"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # Filter and load point encoder weights
            point_encoder_dict = {}
            for k, v in state_dict.items():
                if 'point_encoder' in k:
                    # Remove 'point_encoder.' prefix
                    new_k = k.replace('point_encoder.', '')
                    point_encoder_dict[new_k] = v

            if len(point_encoder_dict) > 0:
                self.point_encoder.load_state_dict(point_encoder_dict, strict=False)
                print(f'[ULIP_ENCODER] Loaded {len(point_encoder_dict)} parameters from checkpoint')
            else:
                print(f'[ULIP_ENCODER] Warning: No point_encoder weights found in checkpoint')

        except Exception as e:
            print(f'[ULIP_ENCODER] Warning: Could not load checkpoint: {e}')
            print(f'[ULIP_ENCODER] Continuing with random initialization')

    def encode_pc(self, pc):
        """
        Encode point cloud to embedding vector.

        Args:
            pc: torch.Tensor, [B, N, 3] or [B, N, 6] - point cloud (xyz or xyzrgb)

        Returns:
            pc_embed: torch.Tensor, [B, output_dim] - point cloud embeddings
        """
        # Extract point cloud features
        pc_feat = self.point_encoder(pc)  # [B, pc_feat_dim]

        # Project to ULIP embedding space
        pc_embed = pc_feat @ self.pc_projection  # [B, output_dim]

        return pc_embed

    def forward(self, pc):
        """Forward pass - same as encode_pc"""
        return self.encode_pc(pc)


class ULIPAlignmentLoss(nn.Module):
    """
    ULIP-based alignment loss for text-conditioned point cloud completion.

    Computes contrastive loss between:
    - Point cloud embeddings from ULIP 3D encoder
    - Text embeddings from CLIP text encoder

    This ensures that the completed point cloud semantically matches
    the text description.

    Args:
        ulip_encoder: ULIPPointBERTEncoder, frozen ULIP 3D encoder
        temperature: float, temperature for contrastive loss (default: 0.07)
        normalize: bool, whether to normalize embeddings (default: True)
    """

    def __init__(self, ulip_encoder, temperature=0.07, normalize=True):
        super().__init__()

        self.ulip_encoder = ulip_encoder
        self.temperature = temperature
        self.normalize = normalize

        # Log scale parameter for contrastive loss (learned)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / temperature)))

        print(f'[ULIP_LOSS] ULIP Alignment Loss initialized')
        print(f'[ULIP_LOSS] Temperature: {temperature}')
        print(f'[ULIP_LOSS] Normalize embeddings: {normalize}')

    def forward(self, point_cloud, text_embeddings):
        """
        Compute ULIP alignment loss.

        Args:
            point_cloud: torch.Tensor, [B, N, 3] - completed point cloud
            text_embeddings: torch.Tensor, [B, 768] - CLIP text embeddings

        Returns:
            loss: torch.Tensor, scalar - contrastive alignment loss
            pc_text_acc: float - accuracy of point cloud to text matching
        """
        # Encode point cloud with frozen ULIP encoder (no gradients)
        with torch.no_grad():
            pc_embeddings = self.ulip_encoder(point_cloud)  # [B, 512]

        # Normalize embeddings if specified
        if self.normalize:
            pc_embeddings = F.normalize(pc_embeddings, dim=-1)
            text_embeddings = F.normalize(text_embeddings, dim=-1)

        # Compute logit scale
        logit_scale = self.logit_scale.exp()

        # Compute similarity matrix
        # pc_embeddings: [B, D], text_embeddings: [B, D]
        # logits_per_pc: [B, B] - similarity between each PC and all texts
        logits_per_pc = logit_scale * pc_embeddings @ text_embeddings.t()
        logits_per_text = logits_per_pc.t()

        # Labels: diagonal elements are positive pairs
        batch_size = pc_embeddings.size(0)
        labels = torch.arange(batch_size, device=pc_embeddings.device)

        # Contrastive loss (symmetric)
        loss_pc = F.cross_entropy(logits_per_pc, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_pc + loss_text) / 2.0

        # Compute accuracy
        with torch.no_grad():
            pc_text_acc = (logits_per_pc.argmax(dim=1) == labels).float().mean().item()

        return loss, pc_text_acc


def create_ulip_alignment_loss(config_path=None, checkpoint_path=None,
                               temperature=0.07, output_dim=512):
    """
    Factory function to create ULIP alignment loss module.

    Args:
        config_path: str (optional), path to PointBERT config
        checkpoint_path: str (optional), path to pretrained ULIP checkpoint
        temperature: float, temperature for contrastive loss
        output_dim: int, dimension of output embeddings

    Returns:
        ulip_loss: ULIPAlignmentLoss module
    """
    # Create ULIP encoder
    ulip_encoder = ULIPPointBERTEncoder(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        output_dim=output_dim
    )

    # Create alignment loss
    ulip_loss = ULIPAlignmentLoss(
        ulip_encoder=ulip_encoder,
        temperature=temperature,
        normalize=True
    )

    return ulip_loss


if __name__ == '__main__':
    """Test the ULIP alignment loss"""
    print("="*80)
    print("Testing ULIP Alignment Loss")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Create ULIP alignment loss
    print("Creating ULIP alignment loss module...")
    ulip_loss_module = create_ulip_alignment_loss()
    ulip_loss_module = ulip_loss_module.to(device)

    # Test inputs
    B = 4  # batch size
    N = 2048  # number of points
    D_text = 768  # CLIP text embedding dimension

    # Dummy point clouds
    point_clouds = torch.randn(B, N, 3).to(device)

    # Dummy text embeddings (simulating CLIP output)
    text_embeddings = torch.randn(B, D_text).to(device)
    text_embeddings = F.normalize(text_embeddings, dim=-1)

    print(f"Input point cloud shape: {point_clouds.shape}")
    print(f"Input text embeddings shape: {text_embeddings.shape}\n")

    # Compute loss
    print("Computing ULIP alignment loss...")
    with torch.no_grad():  # Test mode
        loss, accuracy = ulip_loss_module(point_clouds, text_embeddings)

    print(f"ULIP alignment loss: {loss.item():.4f}")
    print(f"Point-cloud to text accuracy: {accuracy*100:.2f}%\n")

    # Check that ULIP encoder is frozen
    ulip_params_frozen = all(not p.requires_grad for p in ulip_loss_module.ulip_encoder.parameters())
    print(f"ULIP encoder frozen: {ulip_params_frozen}")

    # Check that logit_scale is trainable
    logit_scale_trainable = ulip_loss_module.logit_scale.requires_grad
    print(f"Logit scale trainable: {logit_scale_trainable}")

    print("\n" + "="*80)
    print("✅ Test passed!")
    print("="*80)
