# verify_installation.py
import sys
import subprocess
import pkg_resources

required_packages = [
    ('torch', '2.0.0'),
    ('transformers', '4.30.0'),
    ('datasets', '2.14.0'),
    ('nltk', '3.8.0'),
    ('numpy', '1.24.0'),
    ('pandas', '2.0.0'),
    ('requests', '2.31.0'),
]

print("Verifying MCT project dependencies...")
print("=" * 50)

all_ok = True

for package, min_version in required_packages:
    try:
        installed_version = pkg_resources.get_distribution(package).version
        if pkg_resources.parse_version(installed_version) >= pkg_resources.parse_version(min_version):
            print(f"✓ {package} {installed_version} >= {min_version}")
        else:
            print(f"✗ {package} {installed_version} < {min_version}")
            all_ok = False
    except pkg_resources.DistributionNotFound:
        print(f"✗ {package} not installed")
        all_ok = False

print("=" * 50)

if all_ok:
    print("All dependencies satisfied! ✅")
    
    # Test NLTK data
    try:
        import nltk
        nltk.data.find('corpora/wordnet')
        print("NLTK WordNet data found ✅")
    except LookupError:
        print("NLTK WordNet data not found. Downloading...")
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        print("NLTK data downloaded ✅")
    
    print("\nYou're ready to run the MCT project!")
    
else:
    print("Some dependencies are missing or outdated ❌")
    print("\nTo fix this, run:")
    print("pip install -r requirements.txt")
    sys.exit(1)
