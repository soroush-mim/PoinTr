"""
Test script for text-conditioned AdaPoinTr implementation.

This script tests:
1. Text encoder (CLIPTextEncoder) loads correctly and is frozen
2. Text features have correct shapes
3. Query generator produces correct output shapes
4. AdaPoinTr forward pass works with and without captions
5. Gradient flow (text encoder frozen, other parts trainable)
6. Memory usage is reasonable
7. Backward compatibility
"""

import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.text_encoder import CLIPTextEncoder
from models.text_query_generator import TextConditionedQueryGenerator
from models.AdaPoinTr import AdaPoinTr
from easydict import EasyDict


def test_text_encoder():
    """Test 1: Text encoder loads correctly and is frozen"""
    print("\n" + "="*80)
    print("TEST 1: Text Encoder Loading and Freezing")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize encoder
    encoder = CLIPTextEncoder().to(device)

    # Check all parameters are frozen
    frozen_count = sum(1 for p in encoder.text_model.parameters() if not p.requires_grad)
    total_count = sum(1 for p in encoder.text_model.parameters())

    print(f"Frozen parameters: {frozen_count}/{total_count}")
    assert frozen_count == total_count, "Not all text encoder parameters are frozen!"

    # Check model is in eval mode
    assert not encoder.text_model.training, "Text encoder should be in eval mode!"

    print("✅ PASSED: Text encoder is properly frozen and in eval mode")
    return encoder


def test_text_features(encoder):
    """Test 2: Text features have correct shapes"""
    print("\n" + "="*80)
    print("TEST 2: Text Feature Shapes")
    print("="*80)

    device = next(encoder.text_model.parameters()).device

    # Test captions
    captions = [
        "a wooden chair with four legs",
        "a red sports car with smooth curves",
        "a modern table with glass top",
        "an airplane with long wings"
    ]

    # Encode text
    text_features, text_pooled = encoder.encode_text(captions)

    print(f"Input batch size: {len(captions)}")
    print(f"Text features shape: {text_features.shape}")
    print(f"Text pooled shape: {text_pooled.shape}")

    # Check shapes
    assert text_features.dim() == 3, "Text features should be 3D"
    assert text_features.size(0) == len(captions), "Batch size mismatch"
    assert text_features.size(2) == 768, "CLIP-Large should have 768-dim features"
    assert text_pooled.shape == (len(captions), 768), "Text pooled shape mismatch"

    # Check no gradients
    assert not text_features.requires_grad, "Text features should not require gradients"
    assert not text_pooled.requires_grad, "Text pooled should not require gradients"

    print("✅ PASSED: Text features have correct shapes and no gradients")
    return text_features, text_pooled


def test_query_generator():
    """Test 3: Query generator produces correct output shapes"""
    print("\n" + "="*80)
    print("TEST 3: Text-Conditioned Query Generator")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Parameters
    B = 4
    N = 256
    encoder_dim = 384
    text_dim = 768
    num_queries = 512
    query_dim = 384

    # Create generator
    generator = TextConditionedQueryGenerator(
        encoder_dim=encoder_dim,
        text_dim=text_dim,
        num_queries=num_queries,
        query_dim=query_dim
    ).to(device)

    # Create dummy inputs
    encoder_output = torch.randn(B, N, encoder_dim).to(device)
    text_pooled = torch.randn(B, text_dim).to(device)

    # Generate queries
    queries = generator(encoder_output, text_pooled)

    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Text pooled shape: {text_pooled.shape}")
    print(f"Generated queries shape: {queries.shape}")

    # Check shape
    assert queries.shape == (B, num_queries, query_dim), "Query shape mismatch!"

    print("✅ PASSED: Query generator produces correct output shape")
    return generator


def test_adapointr_forward():
    """Test 4: AdaPoinTr forward pass with and without captions"""
    print("\n" + "="*80)
    print("TEST 4: AdaPoinTr Forward Pass")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create minimal config
    config = EasyDict({
        'NAME': 'AdaPoinTr',
        'use_text_conditioning': True,
        'text_encoder_name': 'openai/clip-vit-large-patch14',
        'num_query': 512,
        'num_points': 8192,
        'center_num': [256, 128],
        'global_feature_dim': 1024,
        'encoder_type': 'graph',
        'decoder_type': 'fc',
        'encoder_config': EasyDict({
            'embed_dim': 384,
            'depth': 6,
            'num_heads': 6,
            'k': 8,
            'n_group': 2,
            'mlp_ratio': 2.,
            'block_style_list': ['attn-graph', 'attn', 'attn', 'attn', 'attn', 'attn'],
            'combine_style': 'concat',
        }),
        'decoder_config': EasyDict({
            'embed_dim': 384,
            'depth': 8,
            'num_heads': 6,
            'k': 8,
            'n_group': 2,
            'mlp_ratio': 2.,
            'self_attn_block_style_list': ['attn-graph', 'attn', 'attn', 'attn', 'attn', 'attn', 'attn', 'attn'],
            'self_attn_combine_style': 'concat',
            'cross_attn_block_style_list': ['attn-graph', 'attn', 'attn', 'attn', 'attn', 'attn', 'attn', 'attn'],
            'cross_attn_combine_style': 'concat',
        })
    })

    # Create model
    print("Creating AdaPoinTr model with text conditioning...")
    model = AdaPoinTr(config).to(device)
    model.eval()

    # Create dummy input
    B = 2
    N_partial = 2048
    partial_pc = torch.randn(B, N_partial, 3).to(device)

    captions = [
        "a wooden chair",
        "a sports car"
    ]

    print(f"\nInput partial point cloud shape: {partial_pc.shape}")

    # Test with captions
    print("\n--- Testing WITH captions ---")
    with torch.no_grad():
        ret_with_captions = model(partial_pc, captions=captions)

    print(f"Output coarse shape: {ret_with_captions[0].shape}")
    print(f"Output fine shape: {ret_with_captions[1].shape}")

    # Test without captions (backward compatibility)
    print("\n--- Testing WITHOUT captions (backward compatibility) ---")
    with torch.no_grad():
        ret_without_captions = model(partial_pc, captions=None)

    print(f"Output coarse shape: {ret_without_captions[0].shape}")
    print(f"Output fine shape: {ret_without_captions[1].shape}")

    # Check both produce valid outputs
    assert ret_with_captions[0].shape == ret_without_captions[0].shape, "Output shapes should match"
    assert not torch.isnan(ret_with_captions[0]).any(), "NaN detected in output with captions"
    assert not torch.isnan(ret_without_captions[0]).any(), "NaN detected in output without captions"

    print("✅ PASSED: Forward pass works with and without captions")
    return model


def test_gradient_flow():
    """Test 5: Gradient flow (text encoder frozen, other parts trainable)"""
    print("\n" + "="*80)
    print("TEST 5: Gradient Flow")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create minimal model
    config = EasyDict({
        'NAME': 'AdaPoinTr',
        'use_text_conditioning': True,
        'text_encoder_name': 'openai/clip-vit-large-patch14',
        'num_query': 256,  # Smaller for faster testing
        'num_points': 8192,
        'center_num': [128, 64],
        'global_feature_dim': 1024,
        'encoder_type': 'graph',
        'decoder_type': 'fc',
        'encoder_config': EasyDict({
            'embed_dim': 384,
            'depth': 2,  # Reduced depth
            'num_heads': 6,
            'k': 8,
            'n_group': 2,
            'mlp_ratio': 2.,
            'block_style_list': ['attn-graph', 'attn'],
            'combine_style': 'concat',
        }),
        'decoder_config': EasyDict({
            'embed_dim': 384,
            'depth': 2,  # Reduced depth
            'num_heads': 6,
            'k': 8,
            'n_group': 2,
            'mlp_ratio': 2.,
            'self_attn_block_style_list': ['attn-graph', 'attn'],
            'self_attn_combine_style': 'concat',
            'cross_attn_block_style_list': ['attn-graph', 'attn'],
            'cross_attn_combine_style': 'concat',
        })
    })

    model = AdaPoinTr(config).to(device)
    model.train()

    # Check text encoder is frozen
    text_encoder_params = list(model.base_model.text_encoder.text_model.parameters())
    frozen_count = sum(1 for p in text_encoder_params if not p.requires_grad)
    print(f"Text encoder frozen parameters: {frozen_count}/{len(text_encoder_params)}")
    assert frozen_count == len(text_encoder_params), "Text encoder should be completely frozen!"

    # Check other parts are trainable
    other_params = [p for name, p in model.named_parameters()
                    if 'text_encoder.text_model' not in name and p.requires_grad]
    print(f"Trainable parameters (excluding frozen text encoder): {len(other_params)}")
    assert len(other_params) > 0, "Other model parts should be trainable!"

    # Test backward pass
    B = 1
    N_partial = 2048
    partial_pc = torch.randn(B, N_partial, 3).to(device)
    captions = ["a test object"]

    # Forward pass
    ret = model(partial_pc, captions=captions)
    loss = ret[0].sum()  # Dummy loss

    # Backward pass
    loss.backward()

    # Check gradients
    text_encoder_grads = [p.grad for p in text_encoder_params if p.grad is not None]
    other_grads = [p.grad for p in other_params if p.grad is not None]

    print(f"Text encoder parameters with gradients: {len(text_encoder_grads)} (should be 0)")
    print(f"Other parameters with gradients: {len(other_grads)} (should be > 0)")

    assert len(text_encoder_grads) == 0, "Text encoder should not receive gradients!"
    assert len(other_grads) > 0, "Other parts should receive gradients!"

    print("✅ PASSED: Gradient flow is correct (text encoder frozen, others trainable)")


def test_memory_usage():
    """Test 6: Memory usage is reasonable"""
    print("\n" + "="*80)
    print("TEST 6: Memory Usage")
    print("="*80)

    if not torch.cuda.is_available():
        print("⚠️  SKIPPED: CUDA not available for memory testing")
        return

    device = 'cuda'

    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Create model
    config = EasyDict({
        'NAME': 'AdaPoinTr',
        'use_text_conditioning': True,
        'text_encoder_name': 'openai/clip-vit-large-patch14',
        'num_query': 512,
        'num_points': 8192,
        'center_num': [256, 128],
        'global_feature_dim': 1024,
        'encoder_type': 'graph',
        'decoder_type': 'fc',
        'encoder_config': EasyDict({
            'embed_dim': 384,
            'depth': 6,
            'num_heads': 6,
            'k': 8,
            'n_group': 2,
            'mlp_ratio': 2.,
            'block_style_list': ['attn-graph', 'attn', 'attn', 'attn', 'attn', 'attn'],
            'combine_style': 'concat',
        }),
        'decoder_config': EasyDict({
            'embed_dim': 384,
            'depth': 8,
            'num_heads': 6,
            'k': 8,
            'n_group': 2,
            'mlp_ratio': 2.,
            'self_attn_block_style_list': ['attn-graph', 'attn', 'attn', 'attn', 'attn', 'attn', 'attn', 'attn'],
            'self_attn_combine_style': 'concat',
            'cross_attn_block_style_list': ['attn-graph', 'attn', 'attn', 'attn', 'attn', 'attn', 'attn', 'attn'],
            'cross_attn_combine_style': 'concat',
        })
    })

    model = AdaPoinTr(config).to(device)
    model.eval()

    memory_after_model = torch.cuda.max_memory_allocated() / 1024**3  # GB
    print(f"Memory after model loading: {memory_after_model:.2f} GB")

    # Test inference
    B = 4
    N_partial = 2048
    partial_pc = torch.randn(B, N_partial, 3).to(device)
    captions = ["test caption"] * B

    with torch.no_grad():
        ret = model(partial_pc, captions=captions)

    memory_after_inference = torch.cuda.max_memory_allocated() / 1024**3  # GB
    print(f"Memory after inference: {memory_after_inference:.2f} GB")
    print(f"Memory increase during inference: {memory_after_inference - memory_after_model:.2f} GB")

    # Check memory is reasonable (should be less than 10GB for batch size 4)
    assert memory_after_inference < 10.0, "Memory usage is too high!"

    print("✅ PASSED: Memory usage is reasonable")


def test_backward_compatibility():
    """Test 7: Backward compatibility (model works without text conditioning)"""
    print("\n" + "="*80)
    print("TEST 7: Backward Compatibility")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create model WITHOUT text conditioning
    config = EasyDict({
        'NAME': 'AdaPoinTr',
        'use_text_conditioning': False,  # Disabled
        'num_query': 512,
        'num_points': 8192,
        'center_num': [256, 128],
        'global_feature_dim': 1024,
        'encoder_type': 'graph',
        'decoder_type': 'fc',
        'encoder_config': EasyDict({
            'embed_dim': 384,
            'depth': 6,
            'num_heads': 6,
            'k': 8,
            'n_group': 2,
            'mlp_ratio': 2.,
            'block_style_list': ['attn-graph', 'attn', 'attn', 'attn', 'attn', 'attn'],
            'combine_style': 'concat',
        }),
        'decoder_config': EasyDict({
            'embed_dim': 384,
            'depth': 8,
            'num_heads': 6,
            'k': 8,
            'n_group': 2,
            'mlp_ratio': 2.,
            'self_attn_block_style_list': ['attn-graph', 'attn', 'attn', 'attn', 'attn', 'attn', 'attn', 'attn'],
            'self_attn_combine_style': 'concat',
            'cross_attn_block_style_list': ['attn-graph', 'attn', 'attn', 'attn', 'attn', 'attn', 'attn', 'attn'],
            'cross_attn_combine_style': 'concat',
        })
    })

    print("Creating model WITHOUT text conditioning...")
    model = AdaPoinTr(config).to(device)
    model.eval()

    # Check text encoder is not initialized
    assert not hasattr(model.base_model, 'text_encoder'), \
        "Text encoder should not exist when text_conditioning=False"

    # Test forward pass
    B = 2
    N_partial = 2048
    partial_pc = torch.randn(B, N_partial, 3).to(device)

    with torch.no_grad():
        ret = model(partial_pc, captions=None)

    print(f"Output coarse shape: {ret[0].shape}")
    print(f"Output fine shape: {ret[1].shape}")

    assert not torch.isnan(ret[0]).any(), "NaN detected in output"

    print("✅ PASSED: Model works correctly without text conditioning")


def run_all_tests():
    """Run all test cases"""
    print("\n")
    print("="*80)
    print("TEXT-CONDITIONED ADAPOINTR - COMPREHENSIVE TEST SUITE")
    print("="*80)

    try:
        # Test 1: Text encoder
        encoder = test_text_encoder()

        # Test 2: Text features
        text_features, text_pooled = test_text_features(encoder)

        # Test 3: Query generator
        query_generator = test_query_generator()

        # Test 4: AdaPoinTr forward pass
        model = test_adapointr_forward()

        # Test 5: Gradient flow
        test_gradient_flow()

        # Test 6: Memory usage
        test_memory_usage()

        # Test 7: Backward compatibility
        test_backward_compatibility()

        # Summary
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        print("\nText-conditioned AdaPoinTr implementation is working correctly.")
        print("\nNext steps:")
        print("1. Prepare caption CSV file (data/shapenet/captions.csv)")
        print("2. Train the model:")
        print("   python main.py --config cfgs/Projected_ShapeNet34_models/AdaPoinTr_text.yaml")
        print("="*80)

        return True

    except Exception as e:
        print("\n" + "="*80)
        print("❌ TEST FAILED!")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
