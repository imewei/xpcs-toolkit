#!/usr/bin/env python3
"""
XPCS Toolkit Test Monitoring and Reporting Script

This script provides comprehensive test monitoring, performance tracking,
and automated reporting capabilities for the XPCS Toolkit test suite.

Features:
- Real-time test execution monitoring
- Performance regression detection
- Coverage trend analysis
- Automated report generation
- Integration with CI/CD systems
"""

import argparse
from datetime import datetime, timedelta
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestMonitor:
    """Monitor and analyze XPCS Toolkit test suite performance."""

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize test monitor.

        Parameters
        ----------
        project_root : Path, optional
            Root directory of the project. If None, auto-detect.
        """
        self.project_root = project_root or Path.cwd()
        self.reports_dir = self.project_root / "test_reports"
        self.reports_dir.mkdir(exist_ok=True)

        # Performance thresholds
        self.performance_thresholds = {
            "regression_threshold": 20.0,  # % increase to flag as regression
            "coverage_minimum": 75.0,  # Minimum acceptable coverage
            "test_timeout": 300,  # Max time for test execution (seconds)
        }

    def run_test_suite(
        self, suite_type: str = "full", save_results: bool = True
    ) -> dict[str, Any]:
        """Run the specified test suite and collect results.

        Parameters
        ----------
        suite_type : str
            Type of test suite to run ('quick', 'core', 'full', 'performance')
        save_results : bool
            Whether to save results to disk

        Returns
        -------
        dict
            Test results summary
        """
        logger.info(f"Running {suite_type} test suite...")

        # Define test commands for different suite types
        test_commands = {
            "quick": [
                "python",
                "-m",
                "pytest",
                "xpcs_toolkit/tests/unit/",
                "--maxfail=5",
                "-x",
                "--tb=short",
            ],
            "core": [
                "python",
                "-m",
                "pytest",
                "xpcs_toolkit/tests/core/",
                "xpcs_toolkit/tests/scientific/",
                "--cov=xpcs_toolkit",
                "--cov-report=xml",
                "--cov-report=json",
            ],
            "full": [
                "python",
                "-m",
                "pytest",
                "xpcs_toolkit/tests/",
                "--cov=xpcs_toolkit",
                "--cov-report=xml",
                "--cov-report=json",
                "--cov-branch",
                "-v",
            ],
            "performance": [
                "python",
                "-m",
                "pytest",
                "xpcs_toolkit/tests/performance/",
                "--benchmark-only",
                "--benchmark-json=benchmark_results.json",
            ],
        }

        if suite_type not in test_commands:
            raise ValueError(f"Unknown test suite type: {suite_type}")

        start_time = time.time()

        try:
            # Run the test command
            result = subprocess.run(
                test_commands[suite_type],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=self.performance_thresholds["test_timeout"],
            )

            end_time = time.time()
            execution_time = end_time - start_time

            # Parse results
            test_results = {
                "suite_type": suite_type,
                "timestamp": datetime.now().isoformat(),
                "execution_time": execution_time,
                "return_code": result.returncode,
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

            # Parse coverage if available
            if suite_type in ["core", "full"]:
                test_results["coverage"] = self._parse_coverage_results()

            # Parse benchmark results if available
            if suite_type == "performance":
                test_results["benchmarks"] = self._parse_benchmark_results()

            # Parse test statistics
            test_results["test_stats"] = self._parse_test_statistics(result.stdout)

            if save_results:
                self._save_test_results(test_results)

            logger.info(f"Test suite completed in {execution_time:.2f}s")
            return test_results

        except subprocess.TimeoutExpired:
            logger.error(
                f"Test suite timed out after {self.performance_thresholds['test_timeout']}s"
            )
            return {
                "suite_type": suite_type,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": "timeout",
                "execution_time": self.performance_thresholds["test_timeout"],
            }

        except Exception as e:
            logger.error(f"Error running test suite: {e}")
            return {
                "suite_type": suite_type,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": str(e),
            }

    def _parse_coverage_results(self) -> dict[str, Any]:
        """Parse coverage results from coverage.json."""
        coverage_file = self.project_root / "coverage.json"

        if not coverage_file.exists():
            return {"error": "Coverage file not found"}

        try:
            with open(coverage_file) as f:
                coverage_data = json.load(f)

            totals = coverage_data.get("totals", {})

            return {
                "line_coverage": totals.get("percent_covered", 0),
                "lines_covered": totals.get("covered_lines", 0),
                "lines_total": totals.get("num_statements", 0),
                "branch_coverage": totals.get("percent_covered_display", "N/A"),
                "missing_lines": totals.get("missing_lines", 0),
            }

        except Exception as e:
            return {"error": f"Failed to parse coverage: {e}"}

    def _parse_benchmark_results(self) -> dict[str, Any]:
        """Parse benchmark results from benchmark_results.json."""
        benchmark_file = self.project_root / "benchmark_results.json"

        if not benchmark_file.exists():
            return {"error": "Benchmark file not found"}

        try:
            with open(benchmark_file) as f:
                benchmark_data = json.load(f)

            benchmarks = benchmark_data.get("benchmarks", [])

            # Extract key metrics
            benchmark_summary = {"total_benchmarks": len(benchmarks), "benchmarks": {}}

            for bench in benchmarks:
                name = bench.get("name", "unknown")
                stats = bench.get("stats", {})
                benchmark_summary["benchmarks"][name] = {
                    "mean": stats.get("mean", 0),
                    "stddev": stats.get("stddev", 0),
                    "median": stats.get("median", 0),
                    "rounds": stats.get("rounds", 0),
                }

            return benchmark_summary

        except Exception as e:
            return {"error": f"Failed to parse benchmarks: {e}"}

    def _parse_test_statistics(self, stdout: str) -> dict[str, Any]:
        """Parse test statistics from pytest output."""
        stats = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0, "warnings": 0}

        # Parse pytest summary line
        lines = stdout.split("\n")
        for line in lines:
            if "passed" in line or "failed" in line:
                # Look for summary lines like "10 passed, 2 failed, 1 skipped"
                if any(
                    word in line for word in ["passed", "failed", "skipped", "error"]
                ):
                    # Basic parsing - could be enhanced
                    if "passed" in line:
                        match = re.search(r"(\d+)\s+passed", line)
                        if match:
                            stats["passed"] = int(match.group(1))

                    if "failed" in line:
                        match = re.search(r"(\d+)\s+failed", line)
                        if match:
                            stats["failed"] = int(match.group(1))

                    if "skipped" in line:
                        match = re.search(r"(\d+)\s+skipped", line)
                        if match:
                            stats["skipped"] = int(match.group(1))

        stats["total"] = (
            stats["passed"] + stats["failed"] + stats["skipped"] + stats["errors"]
        )

        return stats

    def _save_test_results(self, results: dict[str, Any]) -> None:
        """Save test results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results_{results['suite_type']}_{timestamp}.json"
        filepath = self.reports_dir / filename

        try:
            with open(filepath, "w") as f:
                json.dump(results, f, indent=2)

            logger.info(f"Test results saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save test results: {e}")

    def analyze_trends(self, days_back: int = 7) -> dict[str, Any]:
        """Analyze test trends over the specified period.

        Parameters
        ----------
        days_back : int
            Number of days to look back for trend analysis

        Returns
        -------
        dict
            Trend analysis results
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)

        # Find all test result files
        result_files = list(self.reports_dir.glob("test_results_*.json"))

        # Filter by date and load results
        recent_results = []
        for filepath in result_files:
            try:
                # Extract timestamp from filename
                timestamp_str = filepath.stem.split("_")[-2:]  # Last two parts
                timestamp_str = "_".join(timestamp_str)
                file_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")

                if file_date >= cutoff_date:
                    with open(filepath) as f:
                        result_data = json.load(f)
                        result_data["file_date"] = file_date
                        recent_results.append(result_data)

            except Exception as e:
                logger.warning(f"Could not process {filepath}: {e}")

        if not recent_results:
            return {"error": "No recent test results found"}

        # Analyze trends
        trends = {
            "period_days": days_back,
            "total_runs": len(recent_results),
            "success_rate": sum(1 for r in recent_results if r.get("success", False))
            / len(recent_results),
            "avg_execution_time": sum(
                r.get("execution_time", 0) for r in recent_results
            )
            / len(recent_results),
            "coverage_trend": self._analyze_coverage_trend(recent_results),
            "performance_trend": self._analyze_performance_trend(recent_results),
        }

        return trends

    def _analyze_coverage_trend(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze coverage trends."""
        coverage_data = []

        for result in results:
            coverage_info = result.get("coverage", {})
            if "line_coverage" in coverage_info:
                coverage_data.append(
                    {
                        "date": result["file_date"],
                        "coverage": coverage_info["line_coverage"],
                    }
                )

        if not coverage_data:
            return {"error": "No coverage data available"}

        # Sort by date
        coverage_data.sort(key=lambda x: x["date"])

        # Calculate trend
        coverages = [d["coverage"] for d in coverage_data]

        return {
            "current_coverage": coverages[-1] if coverages else 0,
            "average_coverage": sum(coverages) / len(coverages),
            "min_coverage": min(coverages),
            "max_coverage": max(coverages),
            "trend_direction": "up"
            if coverages[-1] > coverages[0]
            else "down"
            if coverages[-1] < coverages[0]
            else "stable",
            "meets_threshold": coverages[-1]
            >= self.performance_thresholds["coverage_minimum"],
        }

    def _analyze_performance_trend(
        self, results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze performance trends."""
        execution_times = []

        for result in results:
            if result.get("success") and "execution_time" in result:
                execution_times.append(
                    {"date": result["file_date"], "time": result["execution_time"]}
                )

        if not execution_times:
            return {"error": "No performance data available"}

        # Sort by date
        execution_times.sort(key=lambda x: x["date"])

        times = [d["time"] for d in execution_times]

        # Check for performance regression
        if len(times) >= 2:
            recent_avg = sum(times[-3:]) / min(3, len(times))  # Last 3 runs
            baseline_avg = sum(times[:3]) / min(3, len(times))  # First 3 runs

            if baseline_avg > 0:
                regression_pct = ((recent_avg - baseline_avg) / baseline_avg) * 100
            else:
                regression_pct = 0
        else:
            regression_pct = 0

        return {
            "current_time": times[-1] if times else 0,
            "average_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "regression_percentage": regression_pct,
            "has_regression": regression_pct
            > self.performance_thresholds["regression_threshold"],
        }

    def generate_report(self, output_format: str = "markdown") -> str:
        """Generate a comprehensive test report.

        Parameters
        ----------
        output_format : str
            Output format ('markdown', 'json', 'html')

        Returns
        -------
        str
            Generated report content
        """
        # Run current test suite
        current_results = self.run_test_suite("core", save_results=True)

        # Analyze trends
        trends = self.analyze_trends(days_back=7)

        if output_format == "markdown":
            return self._generate_markdown_report(current_results, trends)
        elif output_format == "json":
            return json.dumps(
                {"current_results": current_results, "trends": trends}, indent=2
            )
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _generate_markdown_report(
        self, current_results: dict[str, Any], trends: dict[str, Any]
    ) -> str:
        """Generate markdown test report."""
        report_lines = [
            "# XPCS Toolkit Test Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Current Test Results",
            f"- Suite Type: {current_results.get('suite_type', 'unknown')}",
            f"- Success: {'✅' if current_results.get('success') else '❌'}",
            f"- Execution Time: {current_results.get('execution_time', 0):.2f}s",
            "",
        ]

        # Add test statistics
        if "test_stats" in current_results:
            stats = current_results["test_stats"]
            report_lines.extend(
                [
                    "### Test Statistics",
                    f"- Passed: {stats.get('passed', 0)}",
                    f"- Failed: {stats.get('failed', 0)}",
                    f"- Skipped: {stats.get('skipped', 0)}",
                    f"- Total: {stats.get('total', 0)}",
                    "",
                ]
            )

        # Add coverage information
        if "coverage" in current_results:
            coverage = current_results["coverage"]
            if "error" not in coverage:
                report_lines.extend(
                    [
                        "### Code Coverage",
                        f"- Line Coverage: {coverage.get('line_coverage', 0):.1f}%",
                        f"- Lines Covered: {coverage.get('lines_covered', 0)}/{coverage.get('lines_total', 0)}",
                        f"- Meets Threshold: {'✅' if coverage.get('line_coverage', 0) >= self.performance_thresholds['coverage_minimum'] else '❌'}",
                        "",
                    ]
                )

        # Add trend analysis
        if "error" not in trends:
            report_lines.extend(
                [
                    "## Trend Analysis (7 days)",
                    f"- Total Runs: {trends.get('total_runs', 0)}",
                    f"- Success Rate: {trends.get('success_rate', 0):.1%}",
                    f"- Average Execution Time: {trends.get('avg_execution_time', 0):.2f}s",
                    "",
                ]
            )

            # Coverage trends
            coverage_trend = trends.get("coverage_trend", {})
            if "error" not in coverage_trend:
                direction_emoji = {"up": "📈", "down": "📉", "stable": "➡️"}
                report_lines.extend(
                    [
                        "### Coverage Trends",
                        f"- Current: {coverage_trend.get('current_coverage', 0):.1f}%",
                        f"- Average: {coverage_trend.get('average_coverage', 0):.1f}%",
                        f"- Trend: {direction_emoji.get(coverage_trend.get('trend_direction', 'stable'), '❓')} {coverage_trend.get('trend_direction', 'unknown').title()}",
                        f"- Meets Threshold: {'✅' if coverage_trend.get('meets_threshold') else '❌'}",
                        "",
                    ]
                )

            # Performance trends
            perf_trend = trends.get("performance_trend", {})
            if "error" not in perf_trend:
                report_lines.extend(
                    [
                        "### Performance Trends",
                        f"- Current Time: {perf_trend.get('current_time', 0):.2f}s",
                        f"- Average Time: {perf_trend.get('average_time', 0):.2f}s",
                        f"- Performance Change: {perf_trend.get('regression_percentage', 0):+.1f}%",
                        f"- Has Regression: {'⚠️ YES' if perf_trend.get('has_regression') else '✅ NO'}",
                        "",
                    ]
                )

        # Add recommendations
        report_lines.extend(
            [
                "## Recommendations",
                self._generate_recommendations(current_results, trends),
                "",
            ]
        )

        return "\n".join(report_lines)

    def _generate_recommendations(
        self, current_results: dict[str, Any], trends: dict[str, Any]
    ) -> str:
        """Generate recommendations based on test results."""
        recommendations = []

        # Coverage recommendations
        coverage = current_results.get("coverage", {})
        if "line_coverage" in coverage:
            if (
                coverage["line_coverage"]
                < self.performance_thresholds["coverage_minimum"]
            ):
                recommendations.append(
                    f"- 📊 Increase test coverage (current: {coverage['line_coverage']:.1f}%, target: {self.performance_thresholds['coverage_minimum']:.1f}%)"
                )

        # Performance recommendations
        perf_trend = trends.get("performance_trend", {})
        if perf_trend.get("has_regression"):
            recommendations.append(
                f"- ⚡ Investigate performance regression ({perf_trend.get('regression_percentage', 0):+.1f}% increase)"
            )

        # Test failure recommendations
        if not current_results.get("success"):
            recommendations.append(
                "- 🔧 Fix failing tests before proceeding with development"
            )

        # General recommendations
        if trends.get("success_rate", 1.0) < 0.9:
            recommendations.append(
                "- 🎯 Improve test reliability (success rate below 90%)"
            )

        if not recommendations:
            recommendations.append(
                "- ✅ All metrics look good! Keep up the excellent work."
            )

        return "\n".join(recommendations)


def main():
    """Main CLI interface for test monitoring."""
    parser = argparse.ArgumentParser(description="XPCS Toolkit Test Monitor")
    parser.add_argument(
        "command", choices=["run", "analyze", "report"], help="Command to execute"
    )
    parser.add_argument(
        "--suite",
        choices=["quick", "core", "full", "performance"],
        default="core",
        help="Test suite to run",
    )
    parser.add_argument(
        "--days", type=int, default=7, help="Days to look back for trend analysis"
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Report output format",
    )
    parser.add_argument("--output", type=str, help="Output file path (default: stdout)")

    args = parser.parse_args()

    # Initialize monitor
    monitor = TestMonitor()

    try:
        if args.command == "run":
            results = monitor.run_test_suite(args.suite)
            if results.get("success"):
                logger.info("✅ Test suite completed successfully")
                sys.exit(0)
            else:
                logger.error("❌ Test suite failed")
                sys.exit(1)

        elif args.command == "analyze":
            trends = monitor.analyze_trends(args.days)
            print(json.dumps(trends, indent=2))

        elif args.command == "report":
            report_content = monitor.generate_report(args.format)

            if args.output:
                with open(args.output, "w") as f:
                    f.write(report_content)
                logger.info(f"Report saved to {args.output}")
            else:
                print(report_content)

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
