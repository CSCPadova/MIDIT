"""
Inventario archive/ per capire cosa merita documentazione tesi.
"""
from pathlib import Path
import re

def analyze_file(filepath):
    """Analizza file per capire cosa fa."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Cerca docstring principale
        docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
        docstring = docstring_match.group(1).strip() if docstring_match else "No docstring"
        
        # Conta righe
        lines = len(content.split('\n'))
        
        # Cerca classes
        classes = re.findall(r'class\s+(\w+)', content)
        
        return {
            'docstring': docstring[:200],  # First 200 chars
            'lines': lines,
            'classes': classes,
            'size_kb': filepath.stat().st_size / 1024
        }
    except Exception as e:
        return {'error': str(e)}

# Analyze archive directories
print("=" * 80)
print("📚 ARCHIVE INVENTORY - Diffusion Models")
print("=" * 80)

archive_path = Path("midit/archive/old_diffusion_models")
if archive_path.exists():
    for file in sorted(archive_path.glob("*.py")):
        print(f"\n📄 {file.name}")
        print("-" * 80)
        info = analyze_file(file)
        if 'error' not in info:
            print(f"Lines: {info['lines']}")
            print(f"Size: {info['size_kb']:.1f} KB")
            print(f"Classes: {', '.join(info['classes']) if info['classes'] else 'None'}")
            print(f"Description: {info['docstring'][:100]}...")
        else:
            print(f"Error: {info['error']}")
else:
    print("Directory not found!")

print("\n" + "=" * 80)
print("📚 ARCHIVE INVENTORY - Old Dataloaders")
print("=" * 80)

dataloader_path = Path("midit/archive/old_dataloaders")
if dataloader_path.exists():
    for file in sorted(dataloader_path.glob("*.py")):
        print(f"\n📄 {file.name}")
        print("-" * 80)
        info = analyze_file(file)
        if 'error' not in info:
            print(f"Lines: {info['lines']}")
            print(f"Size: {info['size_kb']:.1f} KB")
            print(f"Classes: {', '.join(info['classes']) if info['classes'] else 'None'}")
            print(f"Description: {info['docstring'][:100]}...")
else:
    print("Directory not found!")

print("\n" + "=" * 80)
print("📊 SCRIPTS INVENTORY - Analysis Scripts")
print("=" * 80)

scripts_path = Path("scripts")
analysis_scripts = []
for pattern in ['*analyze*', '*compare*', '*investigate*', '*explore*']:
    analysis_scripts.extend(scripts_path.glob(pattern))

for script in sorted(set(analysis_scripts)):
    if script.is_file() and script.suffix == '.py':
        print(f"\n📄 {script.name}")
        info = analyze_file(script)
        if 'error' not in info:
            print(f"   Lines: {info['lines']}, Size: {info['size_kb']:.1f} KB")
            if info['docstring'] != "No docstring":
                print(f"   Purpose: {info['docstring'][:80]}...")

print("\n" + "=" * 80)
print("✅ Inventory complete!")
print("=" * 80)
