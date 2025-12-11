"""
Audit completo struttura progetto MIDIT.
Identifica: duplicati, file obsoleti, inconsistenze.
"""
from pathlib import Path
import hashlib
from collections import defaultdict

def get_file_hash(filepath):
    """Calculate MD5 hash of file."""
    try:
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

def find_duplicates(root_dir):
    """Find duplicate files by hash."""
    hashes = defaultdict(list)
    
    for file in Path(root_dir).rglob('*'):
        if file.is_file() and not any(skip in str(file) for skip in ['.git', '__pycache__', '.pyc', 'uv.lock']):
            file_hash = get_file_hash(file)
            if file_hash:
                hashes[file_hash].append(file)
    
    duplicates = {h: files for h, files in hashes.items() if len(files) > 1}
    return duplicates

def check_large_files(root_dir, size_mb=10):
    """Find files larger than size_mb."""
    large_files = []
    
    for file in Path(root_dir).rglob('*'):
        if file.is_file() and not any(skip in str(file) for skip in ['.git', 'logs', 'generated_']):
            size = file.stat().st_size / (1024 * 1024)  # MB
            if size > size_mb:
                large_files.append((file, size))
    
    return sorted(large_files, key=lambda x: x[1], reverse=True)

def check_empty_dirs(root_dir):
    """Find empty directories."""
    empty_dirs = []
    
    for dir in Path(root_dir).rglob('*'):
        if dir.is_dir() and not any(dir.iterdir()):
            # Ignore .git, __pycache__
            if '.git' not in str(dir) and '__pycache__' not in str(dir):
                empty_dirs.append(dir)
    
    return empty_dirs

def analyze_scripts_directory():
    """Analyze scripts for organization."""
    scripts_path = Path('scripts')
    
    all_scripts = list(scripts_path.glob('*.py'))
    test_scripts = [s for s in all_scripts if 'test' in s.name]
    debug_scripts = [s for s in all_scripts if 'debug' in s.name]
    analyze_scripts = [s for s in all_scripts if 'analyze' in s.name]
    generate_scripts = [s for s in all_scripts if 'generate' in s.name]
    other_scripts = [s for s in all_scripts if s not in test_scripts + debug_scripts + analyze_scripts + generate_scripts]
    
    return {
        'total': len(all_scripts),
        'test': len(test_scripts),
        'debug': len(debug_scripts),
        'analyze': len(analyze_scripts),
        'generate': len(generate_scripts),
        'other': len(other_scripts),
        'test_files': [s.name for s in test_scripts],
        'debug_files': [s.name for s in debug_scripts],
    }

def check_core_modules():
    """Check core module organization."""
    issues = []
    
    # Check midit/loss/ - potenziale duplicato
    loss_files = list(Path('midit/loss').glob('*.py'))
    if Path('midit/loss/mlm_loss.py').exists():
        issues.append("⚠️ midit/loss/mlm_loss.py exists - check if redundant with multi_head_output.compute_loss()")
    
    # Check midit/train.py vs train_phase1.py
    if Path('midit/train.py').exists() and Path('midit/train_phase1.py').exists():
        issues.append("⚠️ Both train.py and train_phase1.py exist - check which is used")
    
    return issues

print("=" * 80)
print("📊 MIDIT PROJECT STRUCTURE AUDIT")
print("=" * 80)

# 1. Duplicates
print("\n1️⃣ DUPLICATE FILES:")
print("-" * 80)
duplicates = find_duplicates('.')
if duplicates:
    for hash_val, files in duplicates.items():
        print(f"\n🔄 Duplicate group ({len(files)} files):")
        for file in files:
            print(f"   - {file}")
else:
    print("✅ No duplicates found!")

# 2. Large files
print("\n\n2️⃣ LARGE FILES (>10 MB):")
print("-" * 80)
large_files = check_large_files('.', size_mb=10)
if large_files:
    for file, size in large_files[:10]:  # Top 10
        print(f"   {size:>6.1f} MB - {file}")
else:
    print("✅ No large files!")

# 3. Empty directories
print("\n\n3️⃣ EMPTY DIRECTORIES:")
print("-" * 80)
empty_dirs = check_empty_dirs('.')
if empty_dirs:
    for dir in empty_dirs:
        print(f"   📁 {dir}")
else:
    print("✅ No empty directories!")

# 4. Scripts organization
print("\n\n4️⃣ SCRIPTS DIRECTORY ANALYSIS:")
print("-" * 80)
scripts_info = analyze_scripts_directory()
print(f"   Total scripts: {scripts_info['total']}")
print(f"   Test scripts: {scripts_info['test']}")
print(f"   Debug scripts: {scripts_info['debug']}")
print(f"   Analyze scripts: {scripts_info['analyze']}")
print(f"   Generate scripts: {scripts_info['generate']}")
print(f"   Other scripts: {scripts_info['other']}")

if scripts_info['debug'] > 5:
    print(f"\n⚠️ Many debug scripts ({scripts_info['debug']}), consider moving to scripts/obsolete/:")
    for name in scripts_info['debug_files'][:5]:
        print(f"   - {name}")

# 5. Core modules check
print("\n\n5️⃣ CORE MODULE ISSUES:")
print("-" * 80)
issues = check_core_modules()
if issues:
    for issue in issues:
        print(f"   {issue}")
else:
    print("✅ No issues detected!")

# 6. Generated outputs
print("\n\n6️⃣ GENERATED OUTPUT DIRECTORIES:")
print("-" * 80)
generated_dirs = [d for d in Path('.').iterdir() if d.is_dir() and 'generated' in d.name.lower()]
if generated_dirs:
    print(f"   Found {len(generated_dirs)} generated directories:")
    for dir in generated_dirs:
        size = sum(f.stat().st_size for f in dir.rglob('*') if f.is_file()) / (1024 * 1024)
        print(f"   📁 {dir.name} ({size:.1f} MB)")
    print("\n⚠️ Consider: Keep only best generation, move others to archive/")
else:
    print("✅ No generated directories!")

print("\n" + "=" * 80)
print("✅ AUDIT COMPLETE!")
print("=" * 80)
