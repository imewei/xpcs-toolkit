#!/usr/bin/env python3
"""
Optimized test runner for XPCS Toolkit test suite.

This script demonstrates the various optimization options available for running
the test suite efficiently with different performance profiles.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, description):
    """Run a command and measure execution time."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        execution_time = time.time() - start_time
        print(f"\n✓ Success! Execution time: {execution_time:.2f} seconds")
        return True, execution_time
    except subprocess.CalledProcessError as e:
        execution_time = time.time() - start_time
        print(f"\n✗ Failed! Execution time: {execution_time:.2f} seconds")
        print(f"Exit code: {e.returncode}")
        return False, execution_time


def main():
    """Run optimized test configurations."""
    test_configs = [
        {
            "name": "Quick Unit Tests (parallel, no slow tests)",
            "cmd": [
                sys.executable, "-m", "pytest", 
                "xpcs_toolkit/tests/unit/",
                "-v", "-x", "--tb=short",
                "-m", "not slow",
                "-n", "auto",
                "--durations=5"
            ]
        },
        {
            "name": "Integration Tests (optimized)",
            "cmd": [
                sys.executable, "-m", "pytest",
                "xpcs_toolkit/tests/integration/",
                "-v", "--tb=short",
                "-m", "not slow",
                "-n", "2",  # Limited parallelism for integration tests
                "--durations=5"
            ]
        },
        {
            "name": "Performance Benchmarks (serial, subset)",
            "cmd": [
                sys.executable, "-m", "pytest",
                "xpcs_toolkit/tests/performance/",
                "-v", "--tb=short",
                "-m", "benchmark and not slow",
                "--durations=5"
            ]
        },
        {
            "name": "All Tests (parallel, excluding slow)",
            "cmd": [
                sys.executable, "-m", "pytest",
                "xpcs_toolkit/tests/",
                "-v", "--tb=short",
                "-m", "not slow and not benchmark",
                "-n", "auto",
                "--maxfail=3",
                "--durations=10"
            ]
        },
        {
            "name": "Complete Test Suite (optimized)",
            "cmd": [
                sys.executable, "-m", "pytest",
                "xpcs_toolkit/tests/",
                "-v", "--tb=short",
                "-n", "auto",
                "--maxfail=5",
                "--durations=15"
            ]
        }
    ]
    
    print("XPCS Toolkit Optimized Test Runner")
    print("="*60)
    print("This script demonstrates various optimized test configurations.")
    print("Choose a configuration to run:\n")
    
    for i, config in enumerate(test_configs, 1):
        print(f"{i}. {config['name']}")
        
    print(f"{len(test_configs) + 1}. Run all configurations for comparison")
    print("0. Exit")
    
    try:
        choice = int(input("\nEnter your choice (0-{}): ".format(len(test_configs) + 1)))
        
        if choice == 0:
            print("Exiting...")
            return
            
        elif 1 <= choice <= len(test_configs):
            config = test_configs[choice - 1]
            print(f"\nRunning: {config['name']}")
            success, exec_time = run_command(config["cmd"], config["name"])
            
        elif choice == len(test_configs) + 1:
            print("\nRunning all configurations for comparison...")
            total_time = 0
            results = []
            
            for config in test_configs:
                success, exec_time = run_command(config["cmd"], config["name"])
                results.append((config["name"], success, exec_time))
                total_time += exec_time
                
                # Pause between runs
                time.sleep(2)
            
            print(f"\n{'='*60}")
            print("SUMMARY OF ALL RUNS")
            print('='*60)
            for name, success, exec_time in results:
                status = "✓ PASS" if success else "✗ FAIL"
                print(f"{status} {name}: {exec_time:.2f}s")
            
            print(f"\nTotal execution time: {total_time:.2f} seconds")
            
        else:
            print("Invalid choice!")
            
    except (ValueError, KeyboardInterrupt):
        print("\nExiting...")
        return


if __name__ == "__main__":
    main()