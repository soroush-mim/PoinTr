"""
Loss Functions for ULIP-based Test-Time Refinement

Contains three main losses:
1. Text Similarity Loss: Aligns 3D embedding with text embedding
2. Sticking Loss: Keeps refined points close to initial completion (Chamfer Distance)
3. Smoothness Loss: Encourages local smoothness in the point cloud
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add extensions to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'extensions'))
from chamfer_dist import ChamferDistanceL2


def text_similarity_loss(z_3d, z_text):
    """
    Text-to-3D alignment loss using negative cosine similarity.

    Args:
        z_3d: (B, D) 3D point cloud embeddings
        z_text: (B, D) Text embeddings

    Returns:
        loss: Scalar loss (negative cosine similarity)
    """
    # Normalize embeddings
    z_3d_norm = F.normalize(z_3d, dim=-1)
    z_text_norm = F.normalize(z_text, dim=-1)

    # Cosine similarity
    similarity = (z_3d_norm * z_text_norm).sum(dim=-1)

    # Negative similarity (we want to maximize similarity = minimize negative similarity)
    loss = -similarity.mean()

    return loss


def chamfer_distance(P, P0):
    """
    Chamfer Distance between two point clouds.
    This is the "sticking loss" that keeps refined points close to initial completion.

    Args:
        P: (B, N, 3) Refined point cloud
        P0: (B, N, 3) Initial point cloud (from AdaPoinTr)

    Returns:
        loss: Scalar Chamfer Distance
    """
    chamfer_dist = ChamferDistanceL2()
    return chamfer_dist(P, P0)


def smoothness_loss(P, k=8):
    """
    Smoothness loss that encourages local smoothness in the point cloud.
    Penalizes large deviations from local neighborhood average.

    Args:
        P: (B, N, 3) Point cloud
        k: Number of nearest neighbors to consider

    Returns:
        loss: Scalar smoothness loss
    """
    B, N, _ = P.shape

    # Compute pairwise distances
    # P: (B, N, 3)
    # P_expanded: (B, N, 1, 3)
    # P_transposed: (B, 1, N, 3)
    # distances: (B, N, N)
    P_expanded = P.unsqueeze(2)  # (B, N, 1, 3)
    P_transposed = P.unsqueeze(1)  # (B, 1, N, 3)
    distances = torch.sum((P_expanded - P_transposed) ** 2, dim=-1)  # (B, N, N)

    # Find k nearest neighbors (excluding self)
    # topk returns (values, indices), we add small epsilon to avoid self
    distances_no_self = distances + torch.eye(N, device=P.device).unsqueeze(0) * 1e6
    knn_distances, knn_indices = torch.topk(distances_no_self, k, dim=-1, largest=False)

    # Get k nearest neighbor points
    # knn_indices: (B, N, k)
    batch_indices = torch.arange(B, device=P.device).view(B, 1, 1).expand(B, N, k)
    knn_points = P[batch_indices, knn_indices]  # (B, N, k, 3)

    # Compute local mean
    local_mean = knn_points.mean(dim=2)  # (B, N, 3)

    # Smoothness loss: L2 distance from each point to its local neighborhood mean
    loss = torch.mean((P - local_mean) ** 2)

    return loss


def laplacian_smoothness_loss(P, k=8):
    """
    Alternative smoothness loss using Laplacian smoothing.

    Args:
        P: (B, N, 3) Point cloud
        k: Number of nearest neighbors

    Returns:
        loss: Scalar Laplacian smoothness loss
    """
    B, N, _ = P.shape

    # Compute pairwise distances
    P_expanded = P.unsqueeze(2)  # (B, N, 1, 3)
    P_transposed = P.unsqueeze(1)  # (B, 1, N, 3)
    distances = torch.sum((P_expanded - P_transposed) ** 2, dim=-1)  # (B, N, N)

    # Find k nearest neighbors (excluding self)
    distances_no_self = distances + torch.eye(N, device=P.device).unsqueeze(0) * 1e6
    knn_distances, knn_indices = torch.topk(distances_no_self, k, dim=-1, largest=False)

    # Compute Laplacian coordinates
    batch_indices = torch.arange(B, device=P.device).view(B, 1, 1).expand(B, N, k)
    knn_points = P[batch_indices, knn_indices]  # (B, N, k, 3)

    # Laplacian = point - average of neighbors
    laplacian = P - knn_points.mean(dim=2)  # (B, N, 3)

    # Minimize Laplacian magnitude
    loss = torch.mean(laplacian ** 2)

    return loss


class RefinementLoss(nn.Module):
    """
    Combined loss for ULIP-based test-time refinement.

    L_total = λ_text * L_text + λ_stick * L_stick + λ_smooth * L_smooth

    Where:
    - L_text: Negative cosine similarity between 3D and text embeddings
    - L_stick: Chamfer Distance between refined and initial point clouds
    - L_smooth: Smoothness loss based on local neighborhoods
    """

    def __init__(self, lambda_text=0.5, lambda_stick=2.0, lambda_smooth=0.1, k_neighbors=8):
        super().__init__()
        self.lambda_text = lambda_text
        self.lambda_stick = lambda_stick
        self.lambda_smooth = lambda_smooth
        self.k_neighbors = k_neighbors

    def forward(self, P, P0, z_3d, z_text):
        """
        Compute combined refinement loss.

        Args:
            P: (B, N, 3) Refined point cloud
            P0: (B, N, 3) Initial point cloud
            z_3d: (B, D) 3D embedding of P
            z_text: (B, D) Text embedding

        Returns:
            loss_dict: Dictionary with total loss and individual components
        """
        # Compute individual losses
        L_text = text_similarity_loss(z_3d, z_text)
        L_stick = chamfer_distance(P, P0)
        L_smooth = smoothness_loss(P, k=self.k_neighbors)

        # Weighted combination
        L_total = (
            self.lambda_text * L_text +
            self.lambda_stick * L_stick +
            self.lambda_smooth * L_smooth
        )

        return {
            'total': L_total,
            'text': L_text,
            'sticking': L_stick,
            'smoothness': L_smooth
        }


def test_losses():
    """Test loss functions."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create dummy data
    B, N, D = 2, 1024, 512
    P = torch.randn(B, N, 3, device=device)
    P0 = torch.randn(B, N, 3, device=device)
    z_3d = torch.randn(B, D, device=device)
    z_text = torch.randn(B, D, device=device)

    # Test individual losses
    print("Testing individual losses...")
    L_text = text_similarity_loss(z_3d, z_text)
    print(f"Text similarity loss: {L_text.item():.4f}")

    L_stick = chamfer_distance(P, P0)
    print(f"Sticking loss (CD): {L_stick.item():.4f}")

    L_smooth = smoothness_loss(P, k=8)
    print(f"Smoothness loss: {L_smooth.item():.4f}")

    # Test combined loss
    print("\nTesting combined loss...")
    criterion = RefinementLoss()
    loss_dict = criterion(P, P0, z_3d, z_text)
    print(f"Total loss: {loss_dict['total'].item():.4f}")
    print(f"  - Text: {loss_dict['text'].item():.4f}")
    print(f"  - Sticking: {loss_dict['sticking'].item():.4f}")
    print(f"  - Smoothness: {loss_dict['smoothness'].item():.4f}")


if __name__ == '__main__':
    test_losses()
