"""
ULIP-2 Encoder Loader

Loads frozen ULIP-2 3D and text encoders for test-time refinement.
Requires ULIP models to be copied to PoinTr/ulip_models/ directory.
"""

import torch
import torch.nn as nn
import sys
import os
from collections import OrderedDict


def load_ulip_encoders(checkpoint_path, device='cuda', model_type='ULIP2_PointBERT', model_cache_dir="/data/soroush/adapointr/openclip"):
    """
    Load frozen ULIP-2 encoders from checkpoint.

    Args:
        checkpoint_path: Path to ULIP-2 checkpoint (.pt file)
        device: Device to load models on ('cuda' or 'cpu')
        model_type: Type of ULIP model:
            - 'ULIP2_PointBERT' (default, for ULIP-2 with PointBERT encoder)
            - 'ULIP_PointBERT' (for ULIP-1 with PointBERT)
            - 'ULIP_PN_NEXT' (for ULIP-1 with PointNeXt)
        model_cache_dir: Directory to cache downloaded pretrained models (default: None uses ~/.cache/torch/hub/checkpoints/)

    Returns:
        tuple: (encoder_3d, encoder_text) both frozen and in eval mode
    """
    # Add ulip_models to path
    ulip_models_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ulip_models')

    if not os.path.exists(ulip_models_path):
        print(f"ERROR: ULIP models not found at {ulip_models_path}")
        print("Please copy ULIP models to PoinTr/ulip_models/ directory")
        print("\nInstructions:")
        print("  1. mkdir -p ulip_models")
        print("  2. cp -r ../ULIP/models/ULIP_models.py ulip_models/")
        print("  3. cp -r ../ULIP/models/losses.py ulip_models/")
        print("  4. cp -r ../ULIP/models/pointbert ulip_models/")
        print("  5. mkdir -p ulip_models/utils")
        print("  6. cp ../ULIP/utils/config.py ulip_models/utils/")
        print("  7. touch ulip_models/__init__.py ulip_models/utils/__init__.py")
        print("\nUsing dummy encoders for testing...")
        return create_dummy_encoders(device)

    sys.path.insert(0, ulip_models_path)

    try:
        # Import ULIP models
        from ulip_models import ULIP_models

        print(f"Loading ULIP-2 checkpoint from {checkpoint_path}")

        # Load checkpoint
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint not found at {checkpoint_path}")
            print("Using dummy encoders for testing...")
            return create_dummy_encoders(device)

        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Extract state dict
        state_dict = OrderedDict()
        if 'state_dict' in ckpt:
            for k, v in ckpt['state_dict'].items():
                # Remove 'module.' prefix if present
                state_dict[k.replace('module.', '')] = v
        else:
            # Assume checkpoint is already state_dict
            for k, v in ckpt.items():
                state_dict[k.replace('module.', '')] = v

        # Create model based on type
        print(f"Creating {model_type} model...")

        # Create a minimal args object
        class Args:
            def __init__(self):
                self.evaluate_3d = True  # Skip loading SLIP initialization
                self.use_height = False
                self.model_cache_dir = model_cache_dir  # Custom cache directory for pretrained models

        args = Args()

        # Get model constructor
        if model_type == 'ULIP2_PointBERT':
            model_fn = ULIP_models.ULIP2_PointBERT_Colored
        elif model_type == 'ULIP_PointBERT':
            model_fn = ULIP_models.ULIP_PointBERT
        elif model_type == 'ULIP_PN_NEXT':
            model_fn = ULIP_models.ULIP_PN_NEXT
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Create model
        model = model_fn(args=args)

        # Load weights
        print("Loading checkpoint weights...")
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys:
            print(f"Missing keys: {result.missing_keys[:5]}...")  # Show first 5
        if result.unexpected_keys:
            print(f"Unexpected keys: {result.unexpected_keys[:5]}...")  # Show first 5

        # Move to device
        model = model.to(device)
        model.eval()

        # Extract encoders
        encoder_3d = ULIP3DEncoder(model.point_encoder, model.pc_projection, device)

        if hasattr(model, 'open_clip_model'):
            # ULIP-2 with OpenCLIP
            encoder_text = ULIPTextEncoder(model.open_clip_model, device)
        else:
            # ULIP with custom text encoder
            encoder_text = ULIPLegacyTextEncoder(model, device)

        # Freeze encoders
        for param in encoder_3d.parameters():
            param.requires_grad = False
        for param in encoder_text.parameters():
            param.requires_grad = False

        print(f"✓ Successfully loaded ULIP-2 encoders")
        return encoder_3d, encoder_text

    except Exception as e:
        print(f"Error loading ULIP models: {e}")
        import traceback
        traceback.print_exc()
        print("\nUsing dummy encoders for testing...")
        return create_dummy_encoders(device)


def create_dummy_encoders(device):
    """Create dummy encoders for testing when ULIP models are not available."""
    print("Creating dummy encoders (for testing only)")
    encoder_3d = DummyPointEncoder(device)
    encoder_text = DummyTextEncoder(device)
    return encoder_3d, encoder_text


class ULIP3DEncoder(nn.Module):
    """Wrapper for ULIP 3D point cloud encoder.

    Handles both (B, N, 3) xyz-only and (B, N, 6) xyz+rgb point clouds.
    For xyz-only inputs, automatically pads with RGB=0.4 (neutral gray)
    following ULIP's standard preprocessing.
    """

    def __init__(self, point_encoder, projection, device):
        super().__init__()
        self.point_encoder = point_encoder
        self.projection = projection
        self.device = device

    def forward(self, xyz):
        """
        Args:
            xyz: (B, N, 3) or (B, N, 6) point cloud
                 If (B, N, 3): xyz coordinates only
                 If (B, N, 6): xyz + rgb
        Returns:
            features: (B, D) normalized global features
        """
        B, N, C = xyz.shape

        # Pad with RGB if needed
        if C == 3:
            # Add RGB channels with neutral gray (0.4) as per ULIP standard
            rgb = torch.ones(B, N, 3, device=xyz.device, dtype=xyz.dtype) * 0.4
            xyz = torch.cat([xyz, rgb], dim=-1)  # (B, N, 6)
        elif C != 6:
            raise ValueError(f"Expected input with 3 or 6 channels, got {C}")

        # Encode point cloud
        pc_feat = self.point_encoder(xyz)

        # Project to embedding space
        pc_embed = pc_feat @ self.projection

        # Normalize
        pc_embed = pc_embed / pc_embed.norm(dim=-1, keepdim=True)

        return pc_embed


class ULIPTextEncoder(nn.Module):
    """Wrapper for ULIP-2 text encoder (using OpenCLIP)."""

    def __init__(self, open_clip_model, device):
        super().__init__()
        self.open_clip_model = open_clip_model
        self.device = device

        # Get tokenizer
        try:
            import open_clip
            self.tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
            print(f"OpenCLIP tokenizer loaded: {self.tokenizer}")
        except:
            print("Warning: Could not load OpenCLIP tokenizer")
            self.tokenizer = None

    def forward(self, text):
        """
        Args:
            text: String or list of strings
        Returns:
            features: (B, D) normalized text features
        """
        if isinstance(text, str):
            text = [text]

        # Tokenize
        if self.tokenizer is not None:
            text_tokens = self.tokenizer(text).to(self.device)
        else:
            # Fallback - assume text is already tokenized
            text_tokens = text

        # Encode
        text_embed = self.open_clip_model.encode_text(text_tokens)

        # Normalize
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

        return text_embed


class ULIPLegacyTextEncoder(nn.Module):
    """Wrapper for ULIP text encoder (custom transformer)."""

    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device

    def forward(self, text):
        """
        Args:
            text: String or list of strings
        Returns:
            features: (B, D) normalized text features
        """
        if isinstance(text, str):
            text = [text]

        # TODO: Implement tokenization for legacy ULIP
        # This would require the SimpleTokenizer from ULIP
        text_embed = self.model.encode_text(text)

        # Normalize
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

        return text_embed


class DummyPointEncoder(nn.Module):
    """Dummy 3D point cloud encoder for testing.

    Handles both (B, N, 3) and (B, N, 6) inputs.
    For (B, N, 3), pads with RGB=0.4 for consistency.
    """

    def __init__(self, device, embed_dim=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device
        # Use 6 input dimensions to handle xyz + rgb
        self.encoder = nn.Sequential(
            nn.Linear(6, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        ).to(device)

    def forward(self, xyz):
        """
        Args:
            xyz: (B, N, 3) or (B, N, 6) point cloud
        Returns:
            features: (B, embed_dim) global features
        """
        B, N, C = xyz.shape

        # Pad with RGB if needed
        if C == 3:
            rgb = torch.ones(B, N, 3, device=xyz.device, dtype=xyz.dtype) * 0.4
            xyz = torch.cat([xyz, rgb], dim=-1)
        elif C != 6:
            raise ValueError(f"Expected 3 or 6 channels, got {C}")

        # Simple max pooling over points
        point_features = self.encoder(xyz)  # (B, N, embed_dim)
        global_features = torch.max(point_features, dim=1)[0]  # (B, embed_dim)
        # Normalize
        global_features = global_features / global_features.norm(dim=-1, keepdim=True)
        return global_features


class DummyTextEncoder(nn.Module):
    """Dummy text encoder for testing."""

    def __init__(self, device, embed_dim=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device

    def forward(self, text):
        """
        Args:
            text: List of strings or single string
        Returns:
            features: (B, embed_dim) text features
        """
        if isinstance(text, str):
            text = [text]
        batch_size = len(text)
        # Create random features (for testing only)
        features = torch.randn(batch_size, self.embed_dim, device=self.device)
        # Normalize
        features = features / features.norm(dim=-1, keepdim=True)
        return features


def test_encoders():
    """Test encoder loading."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on device: {device}")

    # Test with dummy checkpoint
    print("\n" + "="*80)
    print("Testing ULIP encoder loading...")
    print("="*80)

    encoder_3d, encoder_text = load_ulip_encoders(
        checkpoint_path='path/to/your/checkpoint.pt',
        device=device,
        model_type='ULIP2_PointBERT'
    )

    # Test forward pass
    print("\nTesting forward pass...")
    xyz = torch.randn(2, 1024, 3).to(device)
    text = ["a 3d point cloud of a chair", "a table"]

    with torch.no_grad():
        z_3d = encoder_3d(xyz)
        z_text = encoder_text(text)

    print(f"✓ 3D encoding shape: {z_3d.shape}")
    print(f"✓ Text encoding shape: {z_text.shape}")
    print(f"✓ Cosine similarity: {(z_3d * z_text).sum(dim=-1)}")

    print("\n" + "="*80)
    print("Test completed!")
    print("="*80)


if __name__ == '__main__':
    test_encoders()
