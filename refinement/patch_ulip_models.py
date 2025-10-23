"""
Patch script to fix ULIP_models.py imports.

This script modifies ULIP_models.py to make imports conditional,
so you don't need to copy all encoder types.
"""

import os
import sys

def patch_ulip_models():
    """Patch ULIP_models.py to handle missing imports gracefully."""

    ulip_models_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'ulip_models',
        'ULIP_models.py'
    )

    if not os.path.exists(ulip_models_path):
        print(f"ERROR: {ulip_models_path} not found!")
        print("Please copy ULIP_models.py first")
        return False

    print(f"Patching {ulip_models_path}...")

    # Read original file
    with open(ulip_models_path, 'r') as f:
        lines = f.readlines()

    # Find and replace the problematic import
    patched_lines = []
    for line in lines:
        # Wrap pointnet2 import in try-except
        if 'from models.pointnet2.pointnet2 import Pointnet2_Ssg' in line:
            patched_lines.append('# Conditional import - patched\n')
            patched_lines.append('try:\n')
            patched_lines.append('    from models.pointnet2.pointnet2 import Pointnet2_Ssg\n')
            patched_lines.append('except ImportError:\n')
            patched_lines.append('    Pointnet2_Ssg = None\n')
        # Wrap dataset import in try-except
        elif 'from data.dataset_3d import' in line:
            patched_lines.append('# Conditional import - patched\n')
            patched_lines.append('try:\n')
            patched_lines.append('    from data.dataset_3d import *\n')
            patched_lines.append('except ImportError:\n')
            patched_lines.append('    pass\n')
        else:
            patched_lines.append(line)

    # Write patched file
    with open(ulip_models_path, 'w') as f:
        f.writelines(patched_lines)

    print("✓ Successfully patched ULIP_models.py")
    print("  - Made pointnet2 import conditional")
    print("  - Made dataset_3d import conditional")
    return True

if __name__ == '__main__':
    success = patch_ulip_models()
    sys.exit(0 if success else 1)
