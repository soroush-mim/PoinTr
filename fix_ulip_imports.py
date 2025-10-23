"""
Fix imports in ulip_models directory.

Changes all imports from 'models.' to 'ulip_models.' and 'from models' to 'from ulip_models'
"""

import os
import re
from pathlib import Path


def fix_imports_in_file(file_path):
    """Fix imports in a single Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Pattern 1: from models.something import ...
        content = re.sub(
            r'from models\.(\w+)',
            r'from ulip_models.\1',
            content
        )

        # Pattern 2: import models.something
        content = re.sub(
            r'import models\.(\w+)',
            r'import ulip_models.\1',
            content
        )

        # Pattern 3: from models import ...
        content = re.sub(
            r'from models import',
            r'from ulip_models import',
            content
        )

        # Only write if something changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def fix_all_imports(ulip_models_dir='ulip_models'):
    """Fix imports in all Python files in ulip_models directory."""

    if not os.path.exists(ulip_models_dir):
        print(f"ERROR: Directory {ulip_models_dir} not found!")
        return

    print(f"Fixing imports in {ulip_models_dir}...")
    print("=" * 80)

    # Find all Python files
    py_files = list(Path(ulip_models_dir).rglob('*.py'))

    fixed_count = 0
    for py_file in py_files:
        if fix_imports_in_file(py_file):
            print(f"✓ Fixed: {py_file}")
            fixed_count += 1

    print("=" * 80)
    print(f"Fixed {fixed_count} file(s) out of {len(py_files)} total Python files")

    if fixed_count > 0:
        print("\n✓ All imports have been updated from 'models.' to 'ulip_models.'")
    else:
        print("\nNo files needed fixing (already correct or no Python files found)")


if __name__ == '__main__':
    fix_all_imports()
