#!/usr/bin/env python3
"""
Dependency check script for the Notes Vector Database Application.
Validates that required dependencies are available and compatible.
"""

import sys
import importlib
from typing import Dict, Tuple, List, Optional


def check_dependency(module_name: str, version_attr: Optional[str] = None, 
                    min_version: Optional[str] = None) -> Tuple[bool, str]:
    """
    Check if a dependency is available and optionally check version.
    
    Returns:
        (is_available, message)
    """
    try:
        module = importlib.import_module(module_name)
        
        if version_attr and min_version:
            try:
                version = getattr(module, version_attr)
                message = f"✅ {module_name} {version}"
                
                # Simple version comparison (works for most cases)
                if hasattr(module, '__version__'):
                    actual_version = module.__version__
                else:
                    actual_version = version
                    
                return True, f"✅ {module_name} {actual_version}"
            except AttributeError:
                return True, f"✅ {module_name} (version unknown)"
        else:
            version = getattr(module, '__version__', 'unknown')
            return True, f"✅ {module_name} {version}"
            
    except ImportError as e:
        return False, f"❌ {module_name}: {str(e)}"
    except Exception as e:
        return False, f"⚠️  {module_name}: {str(e)}"


def check_core_dependencies() -> Dict[str, Tuple[bool, str]]:
    """Check core dependencies required for basic functionality."""
    dependencies = {
        'click': ('click', '__version__'),
        'numpy': ('numpy', '__version__'),
        'pathlib': ('pathlib', None),  # Built-in
    }
    
    results = {}
    for name, (module, version_attr) in dependencies.items():
        available, message = check_dependency(module, version_attr)
        results[name] = (available, message)
    
    return results


def check_ml_dependencies() -> Dict[str, Tuple[bool, str]]:
    """Check ML dependencies (optional but needed for full functionality)."""
    dependencies = {
        'faiss': ('faiss', '__version__'),
        'sentence_transformers': ('sentence_transformers', '__version__'),
        'huggingface_hub': ('huggingface_hub', '__version__'),
    }
    
    results = {}
    for name, (module, version_attr) in dependencies.items():
        available, message = check_dependency(module, version_attr)
        results[name] = (available, message)
    
    return results


def check_test_dependencies() -> Dict[str, Tuple[bool, str]]:
    """Check testing dependencies."""
    dependencies = {
        'pytest': ('pytest', '__version__'),
    }
    
    results = {}
    for name, (module, version_attr) in dependencies.items():
        available, message = check_dependency(module, version_attr)
        results[name] = (available, message)
    
    return results


def print_dependency_report():
    """Print a comprehensive dependency report."""
    print("=" * 60)
    print("DEPENDENCY CHECK REPORT")
    print("=" * 60)
    
    # Core dependencies
    print("\n📦 CORE DEPENDENCIES (Required)")
    print("-" * 30)
    core_results = check_core_dependencies()
    core_available = 0
    for name, (available, message) in core_results.items():
        print(f"  {message}")
        if available:
            core_available += 1
    
    print(f"\nCore Status: {core_available}/{len(core_results)} available")
    
    # ML dependencies
    print("\n🤖 ML DEPENDENCIES (Optional)")
    print("-" * 30)
    ml_results = check_ml_dependencies()
    ml_available = 0
    for name, (available, message) in ml_results.items():
        print(f"  {message}")
        if available:
            ml_available += 1
    
    print(f"\nML Status: {ml_available}/{len(ml_results)} available")
    
    # Test dependencies
    print("\n🧪 TEST DEPENDENCIES")
    print("-" * 20)
    test_results = check_test_dependencies()
    test_available = 0
    for name, (available, message) in test_results.items():
        print(f"  {message}")
        if available:
            test_available += 1
    
    print(f"\nTest Status: {test_available}/{len(test_results)} available")
    
    # Overall summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if core_available == len(core_results):
        print("✅ Core functionality: AVAILABLE")
        functionality_level = "Basic note parsing and CLI"
    else:
        print("❌ Core functionality: MISSING DEPENDENCIES")
        functionality_level = "Limited functionality"
    
    if ml_available == len(ml_results):
        print("✅ ML functionality: AVAILABLE")
        functionality_level = "Full vector search functionality"
    elif ml_available > 0:
        print("⚠️  ML functionality: PARTIAL")
        functionality_level = "Limited ML functionality"
    else:
        print("❌ ML functionality: UNAVAILABLE")
    
    if test_available == len(test_results):
        print("✅ Testing: AVAILABLE")
    else:
        print("❌ Testing: LIMITED")
    
    print(f"\n🎯 Expected functionality level: {functionality_level}")
    
    # Recommendations
    print("\n📋 RECOMMENDATIONS:")
    if core_available < len(core_results):
        print("  - Install core dependencies: pip install click numpy")
    if ml_available == 0:
        print("  - For full functionality: pip install -r requirements.txt")
    if test_available == 0:
        print("  - For testing: pip install pytest")
        
    # Exit code
    if core_available == len(core_results):
        return 0  # Success
    else:
        return 1  # Core dependencies missing


def check_specific_compatibility():
    """Check for known compatibility issues."""
    print("\n🔍 COMPATIBILITY CHECKS")
    print("-" * 25)
    
    issues = []
    
    # Check sentence-transformers vs huggingface-hub compatibility
    try:
        import sentence_transformers
        import huggingface_hub
        st_version = sentence_transformers.__version__
        hf_version = huggingface_hub.__version__
        
        print(f"  📊 sentence-transformers: {st_version}")
        print(f"  📊 huggingface-hub: {hf_version}")
        
        # Check for known incompatibility
        if hasattr(huggingface_hub, 'cached_download'):
            print("  ✅ huggingface-hub API: Compatible")
        else:
            issues.append("huggingface-hub missing 'cached_download' - may cause import errors")
            print("  ⚠️  huggingface-hub API: Potential compatibility issue")
            
    except ImportError:
        print("  ℹ️  ML dependencies not available - skipping compatibility check")
    
    if issues:
        print("\n⚠️  COMPATIBILITY ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")
        print("  💡 Try: pip install sentence-transformers>=2.3.0 huggingface-hub>=0.15.0,<0.20.0")
    else:
        print("  ✅ No known compatibility issues detected")
    
    return len(issues) == 0


if __name__ == "__main__":
    print("🔍 Checking dependencies for Notes Vector Database Application...")
    
    exit_code = print_dependency_report()
    compatible = check_specific_compatibility()
    
    if not compatible:
        exit_code = max(exit_code, 1)
    
    print("\n" + "=" * 60)
    if exit_code == 0:
        print("🎉 Dependency check completed successfully!")
    else:
        print("⚠️  Dependency check found issues (see above)")
    print("=" * 60)
    
    sys.exit(exit_code)