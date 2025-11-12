"""
Text Encoder Module for Text-Conditioned Point Cloud Completion
Uses frozen CLIP text encoder to extract semantic features from captions
Supports both HuggingFace CLIP and OpenCLIP models
"""

import torch
import torch.nn as nn

try:
    from transformers import CLIPTextModel, CLIPTokenizer
    HF_CLIP_AVAILABLE = True
except ImportError:
    HF_CLIP_AVAILABLE = False
    print("[TEXT_ENCODER] HuggingFace transformers not available")

try:
    import open_clip
    OPENCLIP_AVAILABLE = True
except ImportError:
    OPENCLIP_AVAILABLE = False
    print("[TEXT_ENCODER] OpenCLIP not available")


class CLIPTextEncoder(nn.Module):
    """
    CLIP Text Encoder wrapper for extracting text features from captions.

    Supports both HuggingFace CLIP and OpenCLIP models:
    - HuggingFace: 'openai/clip-vit-large-patch14' (768-dim)
    - OpenCLIP: 'ViT-bigG-14' (1280-dim, for ULIP2 compatibility)

    All parameters are frozen to maintain the pre-trained representations.

    Args:
        model_name: str, name of the pre-trained CLIP model
                   - For HuggingFace: 'openai/clip-vit-large-patch14'
                   - For OpenCLIP: 'ViT-bigG-14' (default for ULIP2)

    Note:
        Device is automatically detected from model parameters at runtime.
        Use .to(device) or .cuda() to move the entire model to desired device.
    """

    def __init__(self, model_name='ViT-bigG-14'):
        super().__init__()

        self.model_name = model_name
        self.use_openclip = False

        print(f'[TEXT_ENCODER] Loading CLIP text encoder: {model_name}')

        # Determine which CLIP implementation to use
        if model_name.startswith('openai/') or model_name.startswith('laion/'):
            # HuggingFace CLIP
            if not HF_CLIP_AVAILABLE:
                raise ImportError("HuggingFace transformers not available. Install with: pip install transformers")
            self._init_hf_clip(model_name)
        else:
            # OpenCLIP (for ULIP2 compatibility)
            if not OPENCLIP_AVAILABLE:
                raise ImportError("OpenCLIP not available. Install with: pip install open-clip-torch")
            self._init_openclip(model_name)

        print(f'[TEXT_ENCODER] CLIP text encoder loaded successfully')
        print(f'[TEXT_ENCODER] Text feature dimension: {self.text_dim}')
        print(f'[TEXT_ENCODER] Using OpenCLIP: {self.use_openclip}')

    def _init_hf_clip(self, model_name):
        """Initialize HuggingFace CLIP model (768-dim)"""
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.text_model = CLIPTextModel.from_pretrained(model_name)
        except Exception as e:
            print(f'[TEXT_ENCODER] Error loading HuggingFace CLIP model: {e}')
            raise e

        # Freeze all parameters
        for param in self.text_model.parameters():
            param.requires_grad = False

        # Set to eval mode
        self.text_model.eval()

        # CLIP text feature dimension
        self.text_dim = self.text_model.config.hidden_size  # 768 for CLIP-Large
        self.use_openclip = False

    def _init_openclip(self, model_name):
        """Initialize OpenCLIP model (1280-dim for ViT-bigG-14)"""
        try:
            # Create OpenCLIP model
            # For ULIP2, we use ViT-bigG-14 with laion2b pretrained weights
            pretrained = 'laion2b_s39b_b160k' if model_name == 'ViT-bigG-14' else None

            self.text_model, _, _ = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)

        except Exception as e:
            print(f'[TEXT_ENCODER] Error loading OpenCLIP model: {e}')
            raise e

        # Freeze all parameters
        for param in self.text_model.parameters():
            param.requires_grad = False

        # Set to eval mode
        self.text_model.eval()

        # Get text feature dimension
        # For ViT-bigG-14, this is 1280
        if hasattr(self.text_model, 'text_projection'):
            self.text_dim = self.text_model.text_projection.shape[1]
        else:
            # Fallback: run a dummy forward pass
            with torch.no_grad():
                dummy_tokens = torch.zeros((1, 77), dtype=torch.long)
                dummy_output = self.text_model.encode_text(dummy_tokens)
                self.text_dim = dummy_output.shape[-1]

        self.use_openclip = True

    def encode_text(self, captions):
        """
        Encode text captions to feature vectors.

        Args:
            captions: list of str, text captions (batch_size,)

        Returns:
            text_features: torch.Tensor, sequence embeddings [B, seq_len, dim]
            text_pooled: torch.Tensor, pooled embedding [B, dim]
                        - For HuggingFace: [B, 768]
                        - For OpenCLIP ViT-bigG-14: [B, 1280]
        """
        if captions is None or len(captions) == 0:
            raise ValueError("Captions cannot be None or empty")

        # Get device from model parameters (dynamic device detection)
        device = next(self.text_model.parameters()).device

        if self.use_openclip:
            return self._encode_text_openclip(captions, device)
        else:
            return self._encode_text_hf(captions, device)

    def _encode_text_hf(self, captions, device):
        """Encode text using HuggingFace CLIP"""
        # Tokenize text
        try:
            # Tokenize with padding and truncation
            # max_length=77 is the standard CLIP context length
            tokens = self.tokenizer(
                captions,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors='pt'
            )

            # Move tokens to device
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)

        except Exception as e:
            print(f'[TEXT_ENCODER] Error tokenizing captions: {e}')
            print(f'[TEXT_ENCODER] Captions: {captions}')
            raise e

        # Encode text - no gradients needed
        with torch.no_grad():
            try:
                # Get text model outputs
                outputs = self.text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )

                # Extract features
                # last_hidden_state: [B, seq_len, 768] - all token embeddings
                # pooler_output: [B, 768] - CLS token embedding (pooled)
                text_features = outputs.last_hidden_state  # [B, seq_len, 768]
                text_pooled = outputs.pooler_output  # [B, 768]

            except Exception as e:
                print(f'[TEXT_ENCODER] Error encoding text: {e}')
                raise e

        return text_features, text_pooled

    def _encode_text_openclip(self, captions, device):
        """Encode text using OpenCLIP"""
        try:
            # Tokenize with OpenCLIP tokenizer
            # Returns tensor of shape [B, context_length]
            tokens = self.tokenizer(captions).to(device)

        except Exception as e:
            print(f'[TEXT_ENCODER] Error tokenizing captions: {e}')
            print(f'[TEXT_ENCODER] Captions: {captions}')
            raise e

        # Encode text - no gradients needed
        with torch.no_grad():
            try:
                # OpenCLIP encode_text returns pooled features directly
                # Shape: [B, 1280] for ViT-bigG-14
                text_pooled = self.text_model.encode_text(tokens)

                # For compatibility, we also return sequence features
                # We can get these from the text transformer
                # But for now, we'll just duplicate pooled as "features"
                # since AdaPoinTr only uses pooled embeddings
                text_features = text_pooled.unsqueeze(1)  # [B, 1, 1280]

            except Exception as e:
                print(f'[TEXT_ENCODER] Error encoding text: {e}')
                raise e

        return text_features, text_pooled

    def forward(self, captions):
        """
        Forward pass - same as encode_text for consistency.

        Args:
            captions: list of str, text captions

        Returns:
            text_features: torch.Tensor, sequence embeddings [B, seq_len, 768]
            text_pooled: torch.Tensor, pooled embedding [B, 768]
        """
        return self.encode_text(captions)

    @property
    def dim(self):
        """Return the text feature dimension"""
        return self.text_dim


if __name__ == '__main__':
    # Test the text encoder
    print("Testing CLIPTextEncoder...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create encoder
    encoder = CLIPTextEncoder()
    # Move to device
    encoder = encoder.to(device)

    # Test captions
    captions = [
        "a chair with four legs",
        "a wooden table",
        "a modern sofa"
    ]

    # Encode
    print(f"\nEncoding {len(captions)} captions...")
    text_features, text_pooled = encoder.encode_text(captions)

    print(f"Text features shape: {text_features.shape}")  # Should be [3, seq_len, 768]
    print(f"Text pooled shape: {text_pooled.shape}")  # Should be [3, 768]

    # Check that gradients are not computed
    print(f"\nText features requires_grad: {text_features.requires_grad}")  # Should be False
    print(f"Text pooled requires_grad: {text_pooled.requires_grad}")  # Should be False

    print("\nTest passed!")
