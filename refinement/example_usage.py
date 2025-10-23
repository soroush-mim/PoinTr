"""
Example usage of ULIP-based test-time refinement.

This script demonstrates how to use the refinement module with AdaPoinTr.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from refinement import load_ulip_encoders, ULIPRefinement, RefinementConfig


def example_basic_refinement():
    """Example 1: Basic refinement of a single point cloud."""
    print("=" * 80)
    print("Example 1: Basic Refinement")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Load ULIP-2 encoders (will use dummy encoders if checkpoint not found)
    print("Loading ULIP-2 encoders...")
    encoder_3d, encoder_text = load_ulip_encoders(
        checkpoint_path='dummy_checkpoint.pth',  # Replace with actual checkpoint
        device=device
    )

    # Initialize refinement module with default config
    print("Initializing refinement module...")
    refiner = ULIPRefinement(
        encoder_3d=encoder_3d,
        encoder_text=encoder_text,
        lambda_text=0.5,
        lambda_stick=2.0,
        lambda_smooth=0.1,
        device=device
    )

    # Create dummy point cloud (simulating AdaPoinTr output)
    # In practice, this would be: ret = adapointr_model(partial); P_initial = ret[-1]
    P_initial = torch.randn(1, 2048, 3, device=device)
    text_caption = "a 3d point cloud of a chair"

    # Refine the point cloud
    print(f"\nRefining point cloud with caption: '{text_caption}'")
    P_refined = refiner.refine(
        P_initial,
        text_caption,
        steps=15,
        lr=0.05,
        verbose=True
    )

    print(f"\nInitial shape: {P_initial.shape}")
    print(f"Refined shape: {P_refined.shape}")
    print(f"Mean difference: {(P_refined - P_initial).abs().mean().item():.6f}")


def example_batch_refinement():
    """Example 2: Batch refinement with different captions."""
    print("\n" + "=" * 80)
    print("Example 2: Batch Refinement")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load encoders
    encoder_3d, encoder_text = load_ulip_encoders('dummy_checkpoint.pth', device)

    # Initialize refiner
    refiner = ULIPRefinement(encoder_3d, encoder_text, device=device)

    # Create batch of point clouds
    batch_size = 4
    P_batch = torch.randn(batch_size, 2048, 3, device=device)

    # Different captions for each point cloud
    captions = [
        "a 3d point cloud of a chair",
        "a 3d point cloud of a table",
        "a 3d point cloud of a lamp",
        "a 3d point cloud of a sofa"
    ]

    print(f"\nRefining batch of {batch_size} point clouds...")
    P_refined_batch = refiner.refine_batch(
        P_batch,
        captions,
        steps=10,
        lr=0.05,
        verbose=False
    )

    print(f"Batch shape: {P_refined_batch.shape}")
    for i, caption in enumerate(captions):
        diff = (P_refined_batch[i] - P_batch[i]).abs().mean().item()
        print(f"  Sample {i} ('{caption}'): diff = {diff:.6f}")


def example_text_alignment():
    """Example 3: Measuring text-to-3D alignment."""
    print("\n" + "=" * 80)
    print("Example 3: Text-to-3D Alignment")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load encoders and initialize refiner
    encoder_3d, encoder_text = load_ulip_encoders('dummy_checkpoint.pth', device)
    refiner = ULIPRefinement(encoder_3d, encoder_text, device=device)

    # Create point cloud
    P_initial = torch.randn(1, 2048, 3, device=device)
    text_caption = "a 3d point cloud of a modern chair"

    # Compute alignment before refinement
    similarity_before = refiner.compute_text_alignment(P_initial, text_caption)
    print(f"\nText-to-3D similarity before refinement: {similarity_before.item():.4f}")

    # Refine
    P_refined = refiner.refine(
        P_initial,
        text_caption,
        steps=15,
        lr=0.05,
        verbose=False
    )

    # Compute alignment after refinement
    similarity_after = refiner.compute_text_alignment(P_refined, text_caption)
    print(f"Text-to-3D similarity after refinement:  {similarity_after.item():.4f}")
    print(f"Improvement: {(similarity_after - similarity_before).item():.4f}")

    # Compare before and after
    comparison = refiner.compare_before_after(P_initial, P_refined, text_caption)
    print(f"\nComparison results:")
    print(f"  Before:      {comparison['before'].item():.4f}")
    print(f"  After:       {comparison['after'].item():.4f}")
    print(f"  Improvement: {comparison['improvement'].item():.4f}")


def example_different_configs():
    """Example 4: Different refinement configurations."""
    print("\n" + "=" * 80)
    print("Example 4: Different Refinement Configurations")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load encoders
    encoder_3d, encoder_text = load_ulip_encoders('dummy_checkpoint.pth', device)

    # Test different configurations
    configs = {
        'default': RefinementConfig.default(),
        'aggressive': RefinementConfig.aggressive(),
        'conservative': RefinementConfig.conservative()
    }

    P_initial = torch.randn(1, 2048, 3, device=device)
    text_caption = "a 3d point cloud of a chair"

    results = {}
    for config_name, config in configs.items():
        print(f"\n{config_name.upper()} configuration:")
        print(f"  steps={config.steps}, lr={config.lr}")
        print(f"  λ_text={config.lambda_text}, λ_stick={config.lambda_stick}, λ_smooth={config.lambda_smooth}")

        # Initialize refiner with this config
        refiner = ULIPRefinement(
            encoder_3d, encoder_text,
            lambda_text=config.lambda_text,
            lambda_stick=config.lambda_stick,
            lambda_smooth=config.lambda_smooth,
            device=device
        )

        # Refine
        P_refined = refiner.refine(
            P_initial,
            text_caption,
            steps=config.steps,
            lr=config.lr,
            verbose=False
        )

        # Measure differences
        diff = (P_refined - P_initial).abs().mean().item()
        similarity = refiner.compute_text_alignment(P_refined, text_caption).item()

        results[config_name] = {'diff': diff, 'similarity': similarity}
        print(f"  Mean difference: {diff:.6f}")
        print(f"  Text alignment:  {similarity:.4f}")

    # Compare results
    print("\n" + "-" * 80)
    print("Summary:")
    for config_name, res in results.items():
        print(f"{config_name:12s}: diff={res['diff']:.6f}, similarity={res['similarity']:.4f}")


def example_integration_with_adapointr():
    """Example 5: Integration with AdaPoinTr (pseudo-code)."""
    print("\n" + "=" * 80)
    print("Example 5: Integration with AdaPoinTr")
    print("=" * 80)

    print("""
# This is pseudo-code showing how to integrate with AdaPoinTr

# 1. Load AdaPoinTr model (existing code)
from tools import builder
from utils.config import cfg_from_yaml_file

config = cfg_from_yaml_file('cfgs/PCN_models/AdaPoinTr.yaml')
adapointr_model = builder.model_builder(config.model)
builder.load_model(adapointr_model, 'checkpoints/adapointr.pth')
adapointr_model.cuda().eval()

# 2. Load ULIP-2 encoders and initialize refiner
from refinement import load_ulip_encoders, ULIPRefinement

encoder_3d, encoder_text = load_ulip_encoders('checkpoints/ulip2.pth', 'cuda')
refiner = ULIPRefinement(encoder_3d, encoder_text, device='cuda')

# 3. Run inference with refinement
for taxonomy_id, model_id, data, caption in test_dataloader:
    partial, gt = data
    partial = partial.cuda()

    # AdaPoinTr completion
    with torch.no_grad():
        ret = adapointr_model(partial)
        dense_points = ret[-1]  # Initial completion

    # ULIP refinement
    dense_points_refined = refiner.refine(
        dense_points,
        caption,
        steps=15,
        lr=0.05
    )

    # Evaluate
    metrics_baseline = compute_metrics(dense_points, gt)
    metrics_refined = compute_metrics(dense_points_refined, gt)

    print(f"Baseline: {metrics_baseline}")
    print(f"Refined:  {metrics_refined}")
    """)


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("ULIP-based Test-Time Refinement - Example Usage")
    print("=" * 80)

    try:
        example_basic_refinement()
        example_batch_refinement()
        example_text_alignment()
        example_different_configs()
        example_integration_with_adapointr()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
