#!/usr/bin/env python3
"""
XPCS Toolkit - Advanced Test Runner

This script provides comprehensive test execution capabilities for the XPCS Toolkit
with advanced features including parallel execution, coverage reporting, performance
monitoring, and detailed result analysis.

Features:
- Multiple test discovery modes
- Parallel test execution
- Coverage reporting with HTML output
- Performance benchmarking
- Detailed failure analysis
- CI/CD integration support
- Custom test filtering and selection
- Memory usage monitoring
- Test result caching
- Automatic retry for flaky tests

Usage:
    python run_tests.py [options]
    
Examples:
    python run_tests.py --all                    # Run all tests
    python run_tests.py --unit                   # Run only unit tests
    python run_tests.py --integration            # Run integration tests
    python run_tests.py --coverage               # Run with coverage report
    python run_tests.py --parallel 4             # Run with 4 parallel workers
    python run_tests.py --pattern "test_*xpcs*"  # Run tests matching pattern
    python run_tests.py --benchmark              # Include performance benchmarks
    python run_tests.py --ci                     # CI mode (minimal output)
    python run_tests.py --watch                  # Watch mode (rerun on changes)
"""

import sys
import os
import subprocess
import argparse
import time
import json
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile

# Add the package to Python path
PACKAGE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

try:
    import coverage
    COVERAGE_AVAILABLE = True
except ImportError:
    COVERAGE_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Create dummy psutil module for when it's not available
    class DummyProcess:
        def memory_info(self):
            class MemInfo:
                rss = 0
            return MemInfo()
    
    class DummyPsutil:
        NoSuchProcess = Exception
        AccessDenied = Exception
        
        @staticmethod
        def Process():
            return DummyProcess()
    
    psutil = DummyPsutil()

try:
    from watchdog.observers import Observer  # type: ignore[import-untyped]
    from watchdog.events import FileSystemEventHandler  # type: ignore[import-untyped]
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    # Create dummy classes for when watchdog is not available
    class FileSystemEventHandler:
        def on_modified(self, event):
            pass
    
    class Observer:
        def schedule(self, *args, **kwargs):
            pass
        def start(self):
            pass
        def stop(self):
            pass
        def join(self):
            pass


@dataclass
class TestResult:
    """Container for test execution results."""
    name: str
    status: str  # 'passed', 'failed', 'skipped', 'error'
    duration: float
    output: str
    error: Optional[str] = None
    memory_usage: Optional[float] = None
    retry_count: int = 0


@dataclass
class TestSession:
    """Container for overall test session information."""
    start_time: float
    end_time: float
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    total_duration: float
    coverage_percent: Optional[float] = None
    memory_peak: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed / self.total_tests) * 100


class ColorOutput:
    """ANSI color codes for terminal output."""
    
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
    @classmethod
    def colorize(cls, text: str, color: str) -> str:
        """Apply color to text if stdout is a TTY."""
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            return f"{color}{text}{cls.END}"
        return text
    
    @classmethod
    def success(cls, text: str) -> str:
        return cls.colorize(text, cls.GREEN)
    
    @classmethod
    def error(cls, text: str) -> str:
        return cls.colorize(text, cls.RED)
    
    @classmethod  
    def warning(cls, text: str) -> str:
        return cls.colorize(text, cls.YELLOW)
    
    @classmethod
    def info(cls, text: str) -> str:
        return cls.colorize(text, cls.CYAN)
    
    @classmethod
    def bold(cls, text: str) -> str:
        return cls.colorize(text, cls.BOLD)


class MemoryMonitor:
    """Monitor memory usage during test execution."""
    
    def __init__(self):
        self.peak_memory = 0.0
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start memory monitoring in background thread."""
        if not PSUTIL_AVAILABLE:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> float:
        """Stop monitoring and return peak memory usage."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        return self.peak_memory
    
    def _monitor_loop(self):
        """Memory monitoring loop."""
        process = psutil.Process()
        while self.monitoring:
            try:
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                self.peak_memory = max(self.peak_memory, memory_mb)
                time.sleep(0.1)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break


class TestWatcher(FileSystemEventHandler):  # type: ignore[misc]
    """File system watcher for automatic test rerunning."""
    
    def __init__(self, test_runner):
        self.test_runner = test_runner
        self.last_run = 0
        self.debounce_time = 2.0  # seconds
    
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
        
        # Only watch Python files
        if not event.src_path.endswith('.py'):
            return
        
        # Debounce rapid file changes
        current_time = time.time()
        if current_time - self.last_run < self.debounce_time:
            return
        
        self.last_run = current_time
        print(f"\n{ColorOutput.info('File changed:')} {event.src_path}")
        print(ColorOutput.info("Rerunning tests..."))
        self.test_runner.run_tests()


class XPCSTestRunner:
    """Advanced test runner for XPCS Toolkit."""
    
    def __init__(self, args):
        self.args = args
        self.package_dir = Path(__file__).parent
        self.root_dir = self.package_dir.parent
        self.test_dir = self.package_dir / 'tests'
        self.results: List[TestResult] = []
        self.session: Optional[TestSession] = None
        self.memory_monitor = MemoryMonitor()
        
    def discover_tests(self) -> List[Path]:
        """Discover test files based on selection criteria."""
        test_files = []
        
        if self.args.all:
            # Find all test files
            test_files = list(self.test_dir.glob('test_*.py'))
        elif self.args.unit:
            # Unit tests (not marked as integration)
            test_files = [
                f for f in self.test_dir.glob('test_*.py')
                if 'integration' not in f.name.lower()
            ]
        elif self.args.integration:
            # Integration tests only
            test_files = [
                f for f in self.test_dir.glob('test_*.py')
                if 'integration' in f.name.lower()
            ]
        elif self.args.pattern:
            # Pattern matching
            import fnmatch
            all_tests = list(self.test_dir.glob('test_*.py'))
            test_files = [
                f for f in all_tests
                if fnmatch.fnmatch(f.name, self.args.pattern)
            ]
        else:
            # Default: run main test suites
            priority_tests = [
                'test_xpcs_toolkit.py',
                'test_logging.py',
                'test_analysis_kernel.py',
                'test_data_file_locator.py',
                'test_xpcs_file.py'
            ]
            
            test_files = [
                self.test_dir / test_name
                for test_name in priority_tests
                if (self.test_dir / test_name).exists()
            ]
            
            # Add any other test files not in priority list
            all_tests = set(self.test_dir.glob('test_*.py'))
            priority_set = set(test_files)
            remaining_tests = all_tests - priority_set
            test_files.extend(sorted(remaining_tests))
        
        return sorted(test_files)
    
    def run_single_test(self, test_file: Path) -> TestResult:
        """Run a single test file and return results."""
        print(f"  {ColorOutput.info('Running:')} {test_file.name}")
        
        start_time = time.time()
        memory_monitor = MemoryMonitor()
        memory_monitor.start_monitoring()
        
        # Build pytest command
        cmd = [sys.executable, '-m', 'pytest', str(test_file)]
        
        if self.args.verbose:
            cmd.append('-v')
        else:
            cmd.append('-q')
        
        if self.args.benchmark:
            cmd.extend(['--benchmark-only', '--benchmark-sort=mean'])
        
        # Add coverage if requested
        if self.args.coverage and COVERAGE_AVAILABLE:
            cmd.extend([
                '--cov=xpcs_toolkit',
                '--cov-report=term-missing',
                '--cov-append'
            ])
        
        # Run the test
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.args.timeout,
                cwd=self.root_dir
            )
            
            duration = time.time() - start_time
            peak_memory = memory_monitor.stop_monitoring()
            
            # Determine status
            if result.returncode == 0:
                status = 'passed'
            elif 'FAILED' in result.stdout or 'FAILED' in result.stderr:
                status = 'failed'
            elif 'ERROR' in result.stdout or 'ERROR' in result.stderr:
                status = 'error'
            else:
                status = 'unknown'
            
            # Combine output
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            
            error_msg = None
            if result.returncode != 0:
                error_msg = result.stderr or "Test failed with no error output"
            
            return TestResult(
                name=test_file.name,
                status=status,
                duration=duration,
                output=output,
                error=error_msg,
                memory_usage=peak_memory
            )
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            memory_monitor.stop_monitoring()
            
            return TestResult(
                name=test_file.name,
                status='timeout',
                duration=duration,
                output=f"Test timed out after {self.args.timeout} seconds",
                error="Timeout expired"
            )
        
        except Exception as e:
            duration = time.time() - start_time
            memory_monitor.stop_monitoring()
            
            return TestResult(
                name=test_file.name,
                status='error',
                duration=duration,
                output="",
                error=str(e)
            )
    
    def run_tests_parallel(self, test_files: List[Path]) -> List[TestResult]:
        """Run tests in parallel using thread pool."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.args.parallel) as executor:
            # Submit all test jobs
            future_to_test = {
                executor.submit(self.run_single_test, test_file): test_file
                for test_file in test_files
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_test):
                test_file = future_to_test[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Print immediate feedback
                    if result.status == 'passed':
                        print(f"    {ColorOutput.success('✓')} {result.name} ({result.duration:.2f}s)")
                    elif result.status == 'failed':
                        print(f"    {ColorOutput.error('✗')} {result.name} ({result.duration:.2f}s)")
                    elif result.status == 'skipped':
                        print(f"    {ColorOutput.warning('⊝')} {result.name} ({result.duration:.2f}s)")
                    else:
                        print(f"    {ColorOutput.warning('?')} {result.name} ({result.duration:.2f}s)")
                        
                except Exception as e:
                    error_result = TestResult(
                        name=test_file.name,
                        status='error',
                        duration=0.0,
                        output="",
                        error=str(e)
                    )
                    results.append(error_result)
                    print(f"    {ColorOutput.error('✗')} {test_file.name} (error: {e})")
        
        return results
    
    def run_tests_sequential(self, test_files: List[Path]) -> List[TestResult]:
        """Run tests sequentially."""
        results = []
        
        for test_file in test_files:
            result = self.run_single_test(test_file)
            results.append(result)
            
            # Print immediate feedback
            if result.status == 'passed':
                print(f"    {ColorOutput.success('✓')} {result.name} ({result.duration:.2f}s)")
            elif result.status == 'failed':
                print(f"    {ColorOutput.error('✗')} {result.name} ({result.duration:.2f}s)")
            elif result.status == 'skipped':
                print(f"    {ColorOutput.warning('⊝')} {result.name} ({result.duration:.2f}s)")
            else:
                print(f"    {ColorOutput.warning('?')} {result.name} ({result.duration:.2f}s)")
        
        return results
    
    def generate_coverage_report(self) -> Optional[float]:
        """Generate coverage report and return coverage percentage."""
        if not self.args.coverage or not COVERAGE_AVAILABLE:
            return None
        
        try:
            # Generate HTML coverage report
            cmd = [
                sys.executable, '-m', 'coverage', 'html',
                '--directory', str(self.root_dir / 'htmlcov')
            ]
            
            subprocess.run(cmd, capture_output=True, text=True, cwd=self.root_dir)
            
            # Get coverage percentage
            cmd = [sys.executable, '-m', 'coverage', 'report', '--format=total']
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.root_dir)
            
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
            
        except (subprocess.SubprocessError, ValueError):
            pass
        
        return None
    
    def retry_failed_tests(self, failed_results: List[TestResult]) -> List[TestResult]:
        """Retry failed tests up to max_retries times."""
        if not self.args.retry or self.args.retry <= 0:
            return failed_results
        
        print(f"\n{ColorOutput.info('Retrying failed tests...')}")
        
        retried_results = []
        for result in failed_results:
            if result.status not in ['failed', 'error']:
                continue
            
            test_file = self.test_dir / result.name
            if not test_file.exists():
                retried_results.append(result)
                continue
            
            print(f"  {ColorOutput.warning('Retrying:')} {result.name}")
            
            for attempt in range(self.args.retry):
                retry_result = self.run_single_test(test_file)
                retry_result.retry_count = attempt + 1
                
                if retry_result.status == 'passed':
                    print(f"    {ColorOutput.success('✓')} Passed on retry {attempt + 1}")
                    retried_results.append(retry_result)
                    break
                elif attempt == self.args.retry - 1:
                    print(f"    {ColorOutput.error('✗')} Failed all {self.args.retry} retries")
                    retried_results.append(retry_result)
                else:
                    print(f"    {ColorOutput.warning('⊝')} Retry {attempt + 1} failed")
        
        return retried_results
    
    def print_summary(self):
        """Print test execution summary."""
        if not self.session:
            return
        
        print(f"\n{ColorOutput.bold('Test Summary')}")
        print("=" * 50)
        
        print(f"Total Tests:     {self.session.total_tests}")
        print(f"Passed:          {ColorOutput.success(str(self.session.passed))}")
        print(f"Failed:          {ColorOutput.error(str(self.session.failed))}")
        print(f"Skipped:         {ColorOutput.warning(str(self.session.skipped))}")
        print(f"Errors:          {ColorOutput.error(str(self.session.errors))}")
        print(f"Success Rate:    {self.session.success_rate:.1f}%")
        print(f"Total Duration:  {self.session.total_duration:.2f}s")
        
        if self.session.coverage_percent is not None:
            coverage_color = ColorOutput.success if self.session.coverage_percent >= 80 else ColorOutput.warning
            print(f"Coverage:        {coverage_color(f'{self.session.coverage_percent:.1f}%')}")
        
        if self.session.memory_peak is not None:
            print(f"Peak Memory:     {self.session.memory_peak:.1f} MB")
        
        # Show failed tests
        failed_results = [r for r in self.results if r.status in ['failed', 'error']]
        if failed_results:
            print(f"\n{ColorOutput.error('Failed Tests:')}")
            for result in failed_results:
                print(f"  {ColorOutput.error('✗')} {result.name}")
                if result.error and self.args.verbose:
                    print(f"    Error: {result.error[:200]}...")
        
        # Performance summary
        if self.args.benchmark and self.results:
            print(f"\n{ColorOutput.info('Performance:')}")
            sorted_results = sorted(self.results, key=lambda r: r.duration, reverse=True)
            print(f"Slowest test:    {sorted_results[0].name} ({sorted_results[0].duration:.2f}s)")
            print(f"Fastest test:    {sorted_results[-1].name} ({sorted_results[-1].duration:.2f}s)")
            avg_duration = sum(r.duration for r in self.results) / len(self.results)
            print(f"Average time:    {avg_duration:.2f}s")
    
    def save_results(self):
        """Save test results to JSON file."""
        if not self.args.save_results:
            return
        
        results_data = {
            'session': asdict(self.session) if self.session else None,
            'results': [asdict(result) for result in self.results],
            'timestamp': time.time(),
            'args': vars(self.args)
        }
        
        results_file = self.root_dir / 'test_results.json'
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n{ColorOutput.info('Results saved to:')} {results_file}")
    
    def run_tests(self) -> int:
        """Main test execution method."""
        start_time = time.time()
        
        if not self.args.ci:
            print(f"{ColorOutput.bold('XPCS Toolkit Test Runner')}")
            print("=" * 50)
        
        # Check dependencies
        if not PYTEST_AVAILABLE:
            print(ColorOutput.error("Error: pytest is required but not installed"))
            print("Install with: pip install pytest")
            return 1
        
        if self.args.coverage and not COVERAGE_AVAILABLE:
            print(ColorOutput.warning("Warning: coverage requested but not available"))
            print("Install with: pip install coverage")
        
        # Discover tests
        test_files = self.discover_tests()
        if not test_files:
            print(ColorOutput.warning("No test files found"))
            return 0
        
        if not self.args.ci:
            print(f"\n{ColorOutput.info('Discovered tests:')} {len(test_files)}")
            for test_file in test_files:
                print(f"  {test_file.name}")
        
        # Start memory monitoring
        self.memory_monitor.start_monitoring()
        
        # Run tests
        print(f"\n{ColorOutput.info('Running tests...')}")
        
        if self.args.parallel and self.args.parallel > 1:
            if not self.args.ci:
                print(f"Using {self.args.parallel} parallel workers")
            self.results = self.run_tests_parallel(test_files)
        else:
            self.results = self.run_tests_sequential(test_files)
        
        # Retry failed tests if requested
        if self.args.retry and self.args.retry > 0:
            failed_results = [r for r in self.results if r.status in ['failed', 'error']]
            if failed_results:
                retried_results = self.retry_failed_tests(failed_results)
                
                # Replace original results with retry results
                retry_names = {r.name for r in retried_results}
                self.results = [r for r in self.results if r.name not in retry_names]
                self.results.extend(retried_results)
        
        # Stop memory monitoring
        peak_memory = self.memory_monitor.stop_monitoring()
        
        # Generate coverage report
        coverage_percent = None
        if self.args.coverage:
            print(f"\n{ColorOutput.info('Generating coverage report...')}")
            coverage_percent = self.generate_coverage_report()
            if coverage_percent is not None and not self.args.ci:
                print(f"Coverage: {coverage_percent:.1f}%")
        
        # Create session summary
        end_time = time.time()
        
        passed = len([r for r in self.results if r.status == 'passed'])
        failed = len([r for r in self.results if r.status == 'failed'])
        skipped = len([r for r in self.results if r.status == 'skipped'])
        errors = len([r for r in self.results if r.status == 'error'])
        
        self.session = TestSession(
            start_time=start_time,
            end_time=end_time,
            total_tests=len(self.results),
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            total_duration=end_time - start_time,
            coverage_percent=coverage_percent,
            memory_peak=peak_memory
        )
        
        # Print results
        if not self.args.ci:
            self.print_summary()
        else:
            # Minimal CI output
            status = "PASSED" if failed + errors == 0 else "FAILED"
            print(f"{status}: {passed}/{len(self.results)} tests passed in {self.session.total_duration:.2f}s")
        
        # Save results if requested
        if self.args.save_results:
            self.save_results()
        
        # Return appropriate exit code
        return 0 if failed + errors == 0 else 1
    
    def watch_mode(self):
        """Run tests in watch mode."""
        if not WATCHDOG_AVAILABLE:
            print(ColorOutput.error("Watch mode requires watchdog package"))
            print("Install with: pip install watchdog")
            return 1
        
        print(f"{ColorOutput.info('Starting watch mode...')}")
        print("Watching for file changes. Press Ctrl+C to exit.")
        
        # Initial test run
        self.run_tests()
        
        # Set up file watcher
        event_handler = TestWatcher(self)
        observer = Observer()
        observer.schedule(event_handler, str(self.package_dir), recursive=True)
        observer.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"\n{ColorOutput.info('Stopping watch mode...')}")
            observer.stop()
        
        observer.join()
        return 0


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='XPCS Toolkit Advanced Test Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('\n\n')[1] if __doc__ else None  # Use examples from docstring
    )
    
    # Test selection
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument('--all', action='store_true',
                                help='Run all available tests')
    selection_group.add_argument('--unit', action='store_true',
                                help='Run only unit tests')
    selection_group.add_argument('--integration', action='store_true',
                                help='Run only integration tests')
    selection_group.add_argument('--pattern', type=str,
                                help='Run tests matching glob pattern')
    
    # Execution options
    parser.add_argument('--parallel', type=int, default=1,
                       help='Number of parallel test workers (default: 1)')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Test timeout in seconds (default: 300)')
    parser.add_argument('--retry', type=int, default=0,
                       help='Number of retries for failed tests (default: 0)')
    
    # Output options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--ci', action='store_true',
                       help='CI mode (minimal output)')
    parser.add_argument('--save-results', action='store_true',
                       help='Save results to JSON file')
    
    # Analysis options
    parser.add_argument('--coverage', action='store_true',
                       help='Generate coverage report')
    parser.add_argument('--benchmark', action='store_true',
                       help='Include performance benchmarks')
    
    # Special modes
    parser.add_argument('--watch', action='store_true',
                       help='Watch mode (rerun tests on file changes)')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Set default behavior if no specific selection is made
    if not any([args.all, args.unit, args.integration, args.pattern]):
        # Default to running main test suite
        pass
    
    runner = XPCSTestRunner(args)
    
    if args.watch:
        return runner.watch_mode()
    else:
        return runner.run_tests()


if __name__ == '__main__':
    sys.exit(main())