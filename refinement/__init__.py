"""
ULIP-based Test-Time Refinement Module for AdaPoinTr

This module provides plug-and-play test-time refinement for point cloud completion
using frozen ULIP-2 encoders and text-to-3D alignment.
"""

from .ulip_loader import load_ulip_encoders
from .ulip_refinement import ULIPRefinement
from .losses import chamfer_distance, smoothness_loss, text_similarity_loss

__all__ = [
    'load_ulip_encoders',
    'ULIPRefinement',
    'chamfer_distance',
    'smoothness_loss',
    'text_similarity_loss'
]
