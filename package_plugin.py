#!/usr/bin/env python3
"""
Package Dify plugin into .difypkg file
"""
import os
import shutil
import zipfile
from pathlib import Path
import json

def package_plugin(plugin_dir: str, output_file: str = None):
    """
    Package a Dify plugin directory into a .difypkg file
    
    Args:
        plugin_dir: Path to plugin directory (e.g., './qdrant')
        output_file: Output .difypkg file path (optional, auto-generated if not provided)
    """
    plugin_path = Path(plugin_dir).resolve()
    
    if not plugin_path.exists():
        raise FileNotFoundError(f"Plugin directory not found: {plugin_dir}")
    
    # Read manifest to get plugin name and version
    manifest_path = plugin_path / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.yaml not found in {plugin_dir}")
    
    # Generate output filename
    if not output_file:
        # Try to read name and version from manifest
        try:
            import yaml
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = yaml.safe_load(f)
            plugin_name = manifest.get('name', 'plugin')
            version = manifest.get('version', '0.0.1')
            output_file = f"{plugin_name}-{version}.difypkg"
        except:
            output_file = f"{plugin_path.name}.difypkg"
    
    output_path = Path(output_file).resolve()
    
    # Files/directories to include
    include_patterns = [
        'manifest.yaml',
        'main.py',
        'requirements.txt',
        'provider/',
        'tools/',
        'utils/',
        '_assets/',
        'README.md',  # Include README as usage guide
        'PRIVACY.md',
        'LICENSE',
    ]
    
    # Files/directories to exclude
    exclude_patterns = [
        '__pycache__',
        '*.pyc',
        '.git',
        '.env',
        '*.difypkg',
        '*.md',  # Exclude markdown docs (they're for development, not needed in package)
        'test_*.json',
        'test_*.txt',
        'test_*.py',  # Exclude test scripts
        'COMPLIANCE_REPORT.md',
        'PROGRESS_REPORT_FOR_QDRANT.md',
        'MARKETING_INTRO.md',
        'package_plugin.py',  # Don't include the packaging script itself
        'EXAMPLES/',  # Exclude examples directory
        'CONTRIBUTING.md',  # Exclude contributing guide
    ]
    
    print(f"Packaging plugin from: {plugin_path}")
    print(f"Output file: {output_path}")
    
    # Create zip file
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add files
        for pattern in include_patterns:
            pattern_path = plugin_path / pattern
            
            if pattern_path.exists():
                if pattern_path.is_file():
                    # Add single file
                    arcname = pattern_path.relative_to(plugin_path)
                    zipf.write(pattern_path, arcname)
                    print(f"  Added: {arcname}")
                elif pattern_path.is_dir():
                    # Add directory recursively
                    for file_path in pattern_path.rglob('*'):
                        if file_path.is_file():
                            # Skip excluded files
                            should_exclude = False
                            for exclude in exclude_patterns:
                                if exclude in str(file_path) or file_path.match(exclude):
                                    should_exclude = True
                                    break
                            
                            if not should_exclude:
                                arcname = file_path.relative_to(plugin_path)
                                zipf.write(file_path, arcname)
                                print(f"  Added: {arcname}")
            else:
                print(f"  Warning: {pattern} not found, skipping")
    
    print(f"\n✅ Package created successfully: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.2f} KB")
    
    return output_path

if __name__ == '__main__':
    import sys
    
    plugin_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        package_plugin(plugin_dir, output_file)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)

