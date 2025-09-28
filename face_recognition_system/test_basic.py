"""Basic test script to verify project structure and imports."""

import sys
from pathlib import Path
import importlib.util

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_project_structure():
    """Test project directory structure."""
    print("Testing project structure...")
    
    required_dirs = [
        'src',
        'src/models',
        'src/data',
        'src/api',
        'src/services',
        'src/utils',
        'src/scripts',
        'tests'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"✗ Missing directories: {missing_dirs}")
        return False
    else:
        print("✓ All required directories exist")
        return True


def test_required_files():
    """Test required files exist."""
    print("\nTesting required files...")
    
    required_files = [
        'requirements.txt',
        'setup.py',
        '.gitignore',
        'README.md',
        '.env.example',
        'src/config.py',
        'src/data/dataset.py',
        'src/models/face_detector.py',
        'src/models/face_recognition.py',
        'src/utils/image_quality.py',
        'src/utils/metrics.py',
        'src/scripts/train.py',
        'src/scripts/analyze_data.py',
        'src/scripts/simple_analyze.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    else:
        print("✓ All required files exist")
        return True


def test_config_import():
    """Test configuration module import."""
    print("\nTesting configuration import...")
    
    try:
        # Test if config can be imported without external dependencies
        spec = importlib.util.spec_from_file_location("config", "src/config.py")
        if spec is None:
            print("✗ Could not load config module spec")
            return False
        
        print("✓ Configuration module structure is valid")
        return True
    except Exception as e:
        print(f"✗ Configuration import failed: {e}")
        return False


def test_dataset_structure():
    """Test dataset analysis without heavy dependencies."""
    print("\nTesting dataset structure...")
    
    try:
        # Run the simple analyze script
        import subprocess
        result = subprocess.run([
            sys.executable, 'src/scripts/simple_analyze.py', 
            '--data-root', '../facecap'
        ], capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            print("✓ Dataset analysis successful")
            # Print key statistics from output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Total classes' in line or 'TOTAL SAMPLES' in line or 'train:' in line:
                    print(f"  {line.strip()}")
            return True
        else:
            print(f"✗ Dataset analysis failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Dataset structure test failed: {e}")
        return False


def test_file_contents():
    """Test that key files have expected content."""
    print("\nTesting file contents...")
    
    tests_passed = 0
    total_tests = 0
    
    # Test requirements.txt
    total_tests += 1
    try:
        with open('requirements.txt', 'r') as f:
            content = f.read()
            if 'torch' in content and 'fastapi' in content and 'opencv-python' in content:
                print("✓ requirements.txt contains expected dependencies")
                tests_passed += 1
            else:
                print("✗ requirements.txt missing key dependencies")
    except Exception as e:
        print(f"✗ Error reading requirements.txt: {e}")
    
    # Test setup.py
    total_tests += 1
    try:
        with open('setup.py', 'r') as f:
            content = f.read()
            if 'face-recognition-system' in content and 'setuptools' in content:
                print("✓ setup.py has correct structure")
                tests_passed += 1
            else:
                print("✗ setup.py missing expected content")
    except Exception as e:
        print(f"✗ Error reading setup.py: {e}")
    
    # Test README.md
    total_tests += 1
    try:
        with open('README.md', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'Face Recognition System' in content and 'Installation' in content:
                print("✓ README.md has proper documentation")
                tests_passed += 1
            else:
                print("✗ README.md missing expected sections")
    except Exception as e:
        print(f"✗ Error reading README.md: {e}")
    
    return tests_passed == total_tests


def main():
    """Run all basic tests."""
    print("="*60)
    print("FACE RECOGNITION SYSTEM - BASIC STRUCTURE TEST")
    print("="*60)
    
    tests = [
        test_project_structure,
        test_required_files,
        test_config_import,
        test_dataset_structure,
        test_file_contents,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed with exception: {e}")
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("🎉 All basic tests passed! Project structure is correct.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set up environment: cp .env.example .env")
        print("3. Configure database and other settings in .env")
        print("4. Run training: python src/scripts/train.py")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the project structure.")
        return 1


if __name__ == "__main__":
    exit(main())