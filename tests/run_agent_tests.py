#!/usr/bin/env python3
"""
Test runner for AI Disruption Agent comprehensive test suite.

Usage:
    python tests/run_agent_tests.py [options]

Options:
    --coverage      Run with coverage reporting
    --verbose       Enable verbose output
    --edge-cases    Run only edge case tests
    --performance   Run performance tests
    --integration   Run integration tests only
    --quick         Skip slow integration tests
    --debug         Enable debug mode with PDB on failures
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Run AI Disruption Agent test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--coverage", 
        action="store_true", 
        help="Run with coverage reporting (HTML + terminal)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="Enable verbose test output"
    )
    
    parser.add_argument(
        "--edge-cases",
        action="store_true",
        help="Run only edge case tests"
    )
    
    parser.add_argument(
        "--performance", 
        action="store_true",
        help="Run performance and limit tests"
    )
    
    parser.add_argument(
        "--integration",
        action="store_true", 
        help="Run integration scenario tests only"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip slow integration tests for faster execution"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with PDB on test failures"
    )
    
    parser.add_argument(
        "--test-class",
        help="Run specific test class (e.g., TestAgentInitialization)"
    )
    
    parser.add_argument(
        "--parallel", "-j",
        type=int,
        help="Run tests in parallel with N workers"
    )
    
    args = parser.parse_args()
    
    # Build pytest command
    cmd = ["python", "-m", "pytest", "tests/test_agent_new.py"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    
    # Add coverage reporting
    if args.coverage:
        cmd.extend([
            "--cov=gpt_researcher.agent",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-fail-under=90"
        ])
    
    # Add specific test selections
    if args.edge_cases:
        cmd.append("tests/test_agent_new.py::TestEdgeCases")
    elif args.performance:
        cmd.extend(["-m", "performance"])
    elif args.integration:
        cmd.append("tests/test_agent_new.py::TestIntegrationScenarios")
    elif args.test_class:
        cmd.append(f"tests/test_agent_new.py::{args.test_class}")
    
    # Add quick mode (skip slow tests)
    if args.quick:
        cmd.extend(["-m", "not slow"])
    
    # Add debug mode
    if args.debug:
        cmd.extend(["--pdb", "-s"])
        
    # Add parallel execution
    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])
    
    # Ensure we're in the right directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    print("🧪 Running AI Disruption Agent Test Suite")
    print("=" * 50)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 50)
    
    # Run the tests
    try:
        result = subprocess.run(cmd, check=False)
        
        if result.returncode == 0:
            print("\n✅ All tests passed successfully!")
            if args.coverage:
                print("📊 Coverage report generated in htmlcov/ directory")
        else:
            print(f"\n❌ Tests failed with exit code {result.returncode}")
            sys.exit(result.returncode)
            
    except KeyboardInterrupt:
        print("\n⏹️  Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error running tests: {e}")
        sys.exit(1)

def check_dependencies():
    """Check if required test dependencies are installed."""
    required_packages = [
        "pytest",
        "pytest-mock", 
        "pytest-asyncio",
        "pytest-cov"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required test dependencies:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\nInstall with: pip install " + " ".join(missing_packages))
        sys.exit(1)

def run_preset_suites():
    """Run common preset test suites."""
    presets = {
        "full": {
            "description": "Complete test suite with coverage",
            "cmd": ["--coverage", "--verbose"]
        },
        "quick": {
            "description": "Quick test run without slow integration tests", 
            "cmd": ["--quick", "--verbose"]
        },
        "edge": {
            "description": "Edge case tests only",
            "cmd": ["--edge-cases", "--verbose"]
        },
        "integration": {
            "description": "Integration scenario tests",
            "cmd": ["--integration", "--verbose"]
        }
    }
    
    print("\n🎯 Available test presets:")
    for name, config in presets.items():
        print(f"   {name}: {config['description']}")
    
    choice = input("\nSelect preset (or press Enter for full): ").strip().lower()
    
    if choice in presets:
        print(f"\n🚀 Running {choice} preset...")
        sys.argv = ["run_agent_tests.py"] + presets[choice]["cmd"]
        main()
    elif choice == "":
        print("\n🚀 Running full test suite...")
        sys.argv = ["run_agent_tests.py"] + presets["full"]["cmd"]
        main()
    else:
        print(f"❌ Unknown preset: {choice}")
        sys.exit(1)

if __name__ == "__main__":
    # Check dependencies first
    check_dependencies()
    
    # If no arguments provided, show preset options
    if len(sys.argv) == 1:
        run_preset_suites()
    else:
        main() 