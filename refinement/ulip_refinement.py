"""
ULIP-based Test-Time Refinement for Point Cloud Completion

This module implements plug-and-play test-time refinement that:
1. Takes completed point clouds from AdaPoinTr
2. Refines them using frozen ULIP-2 encoders and text alignment
3. Optimizes only point positions, not model weights

The refinement minimizes:
L = λ_text * L_text + λ_stick * L_stick + λ_smooth * L_smooth

Where:
- L_text: Negative cosine similarity between 3D and text embeddings
- L_stick: Chamfer Distance between refined and initial point clouds
- L_smooth: Local smoothness constraint
"""

import torch
import torch.nn as nn
import torch.optim as optim
from .losses import RefinementLoss, text_similarity_loss, chamfer_distance, smoothness_loss


class ULIPRefinement:
    """
    Test-time refinement using frozen ULIP-2 encoders.

    This class implements a plug-and-play refinement module that can be used
    with any point cloud completion method (like AdaPoinTr) without modifying
    the original model.
    """

    def __init__(
        self,
        encoder_3d,
        encoder_text,
        lambda_text=0.5,
        lambda_stick=2.0,
        lambda_smooth=0.1,
        k_neighbors=8,
        device='cuda'
    ):
        """
        Initialize ULIP refinement module.

        Args:
            encoder_3d: Frozen ULIP-2 3D encoder
            encoder_text: Frozen ULIP-2 text encoder
            lambda_text: Weight for text alignment loss
            lambda_stick: Weight for sticking loss (Chamfer Distance)
            lambda_smooth: Weight for smoothness loss
            k_neighbors: Number of neighbors for smoothness computation
            device: Device to run on
        """
        self.encoder_3d = encoder_3d.eval()
        self.encoder_text = encoder_text.eval()
        self.device = device

        # Freeze encoders
        for param in self.encoder_3d.parameters():
            param.requires_grad = False
        for param in self.encoder_text.parameters():
            param.requires_grad = False

        # Initialize loss function
        self.criterion = RefinementLoss(
            lambda_text=lambda_text,
            lambda_stick=lambda_stick,
            lambda_smooth=lambda_smooth,
            k_neighbors=k_neighbors
        )

        self.lambda_text = lambda_text
        self.lambda_stick = lambda_stick
        self.lambda_smooth = lambda_smooth

    def refine(
        self,
        P0,
        text_caption,
        steps=15,
        lr=0.05,
        verbose=False,
        return_trajectory=False
    ):
        """
        Refine point cloud using text-to-3D alignment.

        Args:
            P0: (B, N, 3) Initial point cloud from AdaPoinTr
            text_caption: Text description(s) - string or list of strings
            steps: Number of refinement steps
            lr: Learning rate for optimization
            verbose: Print loss values during refinement
            return_trajectory: Return full refinement trajectory

        Returns:
            P_refined: (B, N, 3) Refined point cloud
            If return_trajectory: (trajectory, loss_history)
        """
        # Ensure P0 is on correct device and create optimizable copy
        P0 = P0.to(self.device).detach()
        P = P0.clone().requires_grad_(True)

        # Setup optimizer - only optimize point positions
        optimizer = optim.Adam([P], lr=lr)

        # Encode text once (frozen encoder)
        with torch.no_grad():
            z_text = self.encoder_text(text_caption)

        # Storage for trajectory
        trajectory = [P0.detach().cpu()] if return_trajectory else None
        loss_history = [] if return_trajectory else None

        # Refinement loop
        for step in range(steps):
            optimizer.zero_grad()

            # Encode current point cloud
            z_3d = self.encoder_3d(P)

            # Compute losses
            loss_dict = self.criterion(P, P0, z_3d, z_text)
            loss = loss_dict['total']

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Logging
            if verbose and (step % 5 == 0 or step == steps - 1):
                print(f"Step {step:3d}/{steps}: "
                      f"Total={loss_dict['total'].item():.4f}, "
                      f"Text={loss_dict['text'].item():.4f}, "
                      f"Stick={loss_dict['sticking'].item():.4f}, "
                      f"Smooth={loss_dict['smoothness'].item():.4f}")

            # Store trajectory
            if return_trajectory:
                trajectory.append(P.detach().cpu())
                loss_history.append({
                    k: v.item() for k, v in loss_dict.items()
                })

        # Return refined point cloud
        P_refined = P.detach()

        if return_trajectory:
            return P_refined, (trajectory, loss_history)
        else:
            return P_refined

    def refine_batch(
        self,
        P0_batch,
        text_captions,
        steps=15,
        lr=0.05,
        verbose=False
    ):
        """
        Refine a batch of point clouds with corresponding text captions.

        Args:
            P0_batch: (B, N, 3) Batch of initial point clouds
            text_captions: List of B text descriptions
            steps: Number of refinement steps
            lr: Learning rate
            verbose: Print progress

        Returns:
            P_refined_batch: (B, N, 3) Batch of refined point clouds
        """
        return self.refine(
            P0_batch,
            text_captions,
            steps=steps,
            lr=lr,
            verbose=verbose,
            return_trajectory=False
        )

    def compute_text_alignment(self, P, text_caption):
        """
        Compute text-to-3D alignment score (cosine similarity).

        Args:
            P: (B, N, 3) Point cloud
            text_caption: Text description(s)

        Returns:
            similarity: (B,) Cosine similarity scores
        """
        with torch.no_grad():
            P = P.to(self.device)
            z_3d = self.encoder_3d(P)
            z_text = self.encoder_text(text_caption)

            # Normalize
            z_3d_norm = torch.nn.functional.normalize(z_3d, dim=-1)
            z_text_norm = torch.nn.functional.normalize(z_text, dim=-1)

            # Cosine similarity
            similarity = (z_3d_norm * z_text_norm).sum(dim=-1)

        return similarity

    def compare_before_after(self, P0, P_refined, text_caption):
        """
        Compare text alignment before and after refinement.

        Args:
            P0: (B, N, 3) Initial point cloud
            P_refined: (B, N, 3) Refined point cloud
            text_caption: Text description(s)

        Returns:
            dict with 'before', 'after', and 'improvement' scores
        """
        similarity_before = self.compute_text_alignment(P0, text_caption)
        similarity_after = self.compute_text_alignment(P_refined, text_caption)

        return {
            'before': similarity_before.cpu(),
            'after': similarity_after.cpu(),
            'improvement': (similarity_after - similarity_before).cpu()
        }


class RefinementConfig:
    """Configuration for ULIP refinement."""

    def __init__(
        self,
        steps=15,
        lr=0.05,
        lambda_text=0.5,
        lambda_stick=2.0,
        lambda_smooth=0.1,
        k_neighbors=8,
        verbose=False
    ):
        self.steps = steps
        self.lr = lr
        self.lambda_text = lambda_text
        self.lambda_stick = lambda_stick
        self.lambda_smooth = lambda_smooth
        self.k_neighbors = k_neighbors
        self.verbose = verbose

    @classmethod
    def default(cls):
        """Default configuration."""
        return cls()

    @classmethod
    def aggressive(cls):
        """More aggressive refinement (more text weight, more steps)."""
        return cls(
            steps=30,
            lr=0.08,
            lambda_text=1.0,
            lambda_stick=1.5,
            lambda_smooth=0.05
        )

    @classmethod
    def conservative(cls):
        """Conservative refinement (stays closer to initial)."""
        return cls(
            steps=10,
            lr=0.03,
            lambda_text=0.3,
            lambda_stick=3.0,
            lambda_smooth=0.2
        )


def test_refinement():
    """Test refinement module."""
    from .ulip_loader import load_ulip_encoders

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load encoders (will use dummy encoders for testing)
    print("Loading ULIP-2 encoders...")
    encoder_3d, encoder_text = load_ulip_encoders('dummy_checkpoint.pth', device)

    # Initialize refinement module
    print("Initializing refinement module...")
    refiner = ULIPRefinement(
        encoder_3d,
        encoder_text,
        lambda_text=0.5,
        lambda_stick=2.0,
        lambda_smooth=0.1,
        device=device
    )

    # Create dummy data
    B, N = 2, 2048
    P0 = torch.randn(B, N, 3, device=device)
    text_captions = [
        "a 3d point cloud of a chair",
        "a 3d point cloud of a table"
    ]

    # Test refinement
    print(f"\nRefining {B} point clouds with {N} points each...")
    P_refined = refiner.refine(
        P0,
        text_captions,
        steps=15,
        lr=0.05,
        verbose=True
    )

    print(f"\nRefined point cloud shape: {P_refined.shape}")

    # Compare before/after
    print("\nComparing text alignment before and after refinement...")
    comparison = refiner.compare_before_after(P0, P_refined, text_captions)
    print(f"Before: {comparison['before']}")
    print(f"After: {comparison['after']}")
    print(f"Improvement: {comparison['improvement']}")


if __name__ == '__main__':
    test_refinement()
