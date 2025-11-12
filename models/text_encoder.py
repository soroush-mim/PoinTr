"""
Text Encoder Module for Text-Conditioned Point Cloud Completion
Uses frozen CLIP text encoder to extract semantic features from captions
"""

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


class CLIPTextEncoder(nn.Module):
    """
    CLIP Text Encoder wrapper for extracting text features from captions.

    Uses the pre-trained CLIP-ViT-Large-Patch14 model from HuggingFace.
    All parameters are frozen to maintain the pre-trained representations.

    Args:
        model_name: str, name of the pre-trained CLIP model (default: 'openai/clip-vit-large-patch14')

    Note:
        Device is automatically detected from model parameters at runtime.
        Use .to(device) or .cuda() to move the entire model to desired device.
    """

    def __init__(self, model_name='openai/clip-vit-large-patch14'):
        super().__init__()

        print(f'[TEXT_ENCODER] Loading CLIP text encoder: {model_name}')

        # Load pre-trained CLIP text encoder and tokenizer
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
            self.text_model = CLIPTextModel.from_pretrained(model_name)
        except Exception as e:
            print(f'[TEXT_ENCODER] Error loading CLIP model: {e}')
            print(f'[TEXT_ENCODER] Make sure transformers library is installed: pip install transformers')
            raise e

        # Freeze all parameters - we don't want to fine-tune CLIP
        for param in self.text_model.parameters():
            param.requires_grad = False

        # Set to eval mode
        self.text_model.eval()

        # CLIP text feature dimension
        self.text_dim = self.text_model.config.hidden_size  # 768 for CLIP-Large

        print(f'[TEXT_ENCODER] CLIP text encoder loaded successfully')
        print(f'[TEXT_ENCODER] Text feature dimension: {self.text_dim}')
        print(f'[TEXT_ENCODER] All parameters frozen: {not any(p.requires_grad for p in self.text_model.parameters())}')

    def encode_text(self, captions):
        """
        Encode text captions to feature vectors.

        Args:
            captions: list of str, text captions (batch_size,)

        Returns:
            text_features: torch.Tensor, sequence embeddings [B, seq_len, 768]
            text_pooled: torch.Tensor, pooled CLS token embedding [B, 768]
        """
        if captions is None or len(captions) == 0:
            raise ValueError("Captions cannot be None or empty")

        # Get device from model parameters (dynamic device detection)
        device = next(self.text_model.parameters()).device

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
