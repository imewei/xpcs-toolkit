#!/usr/bin/env python3
"""
Test runner for XPCS Toolkit test suite.

This script provides a convenient interface for running different categories of tests
with appropriate configuration and reporting.
"""

from __future__ import annotations

import argparse
import sys
import subprocess
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*50)
    
    try:
        result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent.parent.parent)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run XPCS Toolkit tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --unit                    # Run unit tests only
  %(prog)s --integration             # Run integration tests only  
  %(prog)s --performance             # Run performance tests only
  %(prog)s --fileio                  # Run FileIO tests only
  %(prog)s --all                     # Run all tests
  %(prog)s --fast                    # Run all tests except slow ones
  %(prog)s --coverage                # Run with coverage reporting
  %(prog)s --unit --coverage         # Unit tests with coverage
        """
    )
    
    # Test category options
    parser.add_argument('--unit', action='store_true',
                       help='Run unit tests')
    parser.add_argument('--integration', action='store_true',
                       help='Run integration tests')
    parser.add_argument('--performance', action='store_true', 
                       help='Run performance tests')
    parser.add_argument('--fileio', action='store_true',
                       help='Run FileIO tests')
    parser.add_argument('--all', action='store_true',
                       help='Run all tests')
    parser.add_argument('--fast', action='store_true',
                       help='Run all tests except slow ones')
    
    # Test configuration options
    parser.add_argument('--coverage', action='store_true',
                       help='Generate coverage report')
    parser.add_argument('--html-cov', action='store_true',
                       help='Generate HTML coverage report')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose test output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet test output')
    parser.add_argument('--parallel', '-n', type=int, metavar='N',
                       help='Run tests in parallel with N workers')
    
    args = parser.parse_args()
    
    # Default to running all tests if no specific category is selected
    if not any([args.unit, args.integration, args.performance, args.fileio, args.all, args.fast]):
        args.all = True
    
    base_cmd = ['python', '-m', 'pytest']
    
    # Add verbosity
    if args.verbose:
        base_cmd.append('-v')
    elif args.quiet:
        base_cmd.append('-q')
    
    # Add parallel execution
    if args.parallel:
        base_cmd.extend(['-n', str(args.parallel)])
    
    # Add coverage options
    if args.coverage or args.html_cov:
        base_cmd.extend(['--cov=xpcs_toolkit'])
        if args.html_cov:
            base_cmd.append('--cov-report=html')
        else:
            base_cmd.append('--cov-report=term-missing')
    
    tests_passed = []
    tests_failed = []
    
    # Run unit tests
    if args.unit or args.all:
        cmd = base_cmd + ['xpcs_toolkit/tests/unit/']
        success = run_command(cmd, "Unit Tests")
        (tests_passed if success else tests_failed).append("Unit")
    
    # Run integration tests  
    if args.integration or args.all:
        cmd = base_cmd + ['xpcs_toolkit/tests/integration/']
        success = run_command(cmd, "Integration Tests")
        (tests_passed if success else tests_failed).append("Integration")
    
    # Run performance tests
    if args.performance or args.all:
        cmd = base_cmd + ['xpcs_toolkit/tests/performance/']
        success = run_command(cmd, "Performance Tests") 
        (tests_passed if success else tests_failed).append("Performance")
    
    # Run FileIO tests
    if args.fileio or args.all:
        cmd = base_cmd + ['xpcs_toolkit/tests/fileio/']
        success = run_command(cmd, "FileIO Tests")
        (tests_passed if success else tests_failed).append("FileIO")
    
    # Run fast tests (exclude slow markers)
    if args.fast:
        cmd = base_cmd + ['-m', 'not slow', 'xpcs_toolkit/tests/']
        success = run_command(cmd, "Fast Tests")
        (tests_passed if success else tests_failed).append("Fast")
    
    # Print summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print('='*50)
    
    if tests_passed:
        print(f"✅ PASSED: {', '.join(tests_passed)}")
    if tests_failed:
        print(f"❌ FAILED: {', '.join(tests_failed)}")
    
    total_categories = len(tests_passed) + len(tests_failed)
    if total_categories > 0:
        success_rate = len(tests_passed) / total_categories * 100
        print(f"📊 SUCCESS RATE: {success_rate:.1f}% ({len(tests_passed)}/{total_categories})")
    
    # Exit with appropriate code
    sys.exit(0 if not tests_failed else 1)


if __name__ == '__main__':
    main()