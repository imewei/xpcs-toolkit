"""
Synthetic Data Generation for XPCS Toolkit Testing

This module provides utilities to generate synthetic XPCS data for testing purposes.
It creates realistic datasets with known analytical solutions for correlation functions,
enabling comprehensive numerical accuracy testing.
"""

import os
from pathlib import Path
import tempfile
from typing import Any, Optional

import h5py
import numpy as np


class SyntheticXPCSDataGenerator:
    """
    Generate synthetic XPCS data with known analytical solutions.

    This class creates realistic XPCS datasets for testing numerical accuracy,
    performance benchmarking, and robustness testing.
    """

    def __init__(self, random_seed: int = 42):
        """
        Initialize the synthetic data generator.

        Parameters
        ----------
        random_seed : int, optional
            Random seed for reproducible data generation, by default 42
        """
        np.random.seed(random_seed)
        self.random_seed = random_seed

    def generate_brownian_motion_intensity(
        self,
        n_times: int = 1000,
        n_q_bins: int = 20,
        diffusion_coefficient: float = 1e-12,
        q_values: Optional[np.ndarray] = None,
        noise_level: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic intensity time series for Brownian motion.

        For Brownian motion, the analytical g2 function is:
        g2(q,τ) = 1 + β * exp(-2Γτ) where Γ = D*q²

        Parameters
        ----------
        n_times : int, optional
            Number of time points, by default 1000
        n_q_bins : int, optional
            Number of q-bins, by default 20
        diffusion_coefficient : float, optional
            Diffusion coefficient in m²/s, by default 1e-12
        q_values : np.ndarray, optional
            Q-values in Å⁻¹. If None, will generate logarithmically spaced values
        noise_level : float, optional
            Relative noise level, by default 0.1

        Returns
        -------
        intensity : np.ndarray
            Shape (n_times, n_q_bins) intensity time series
        q_values : np.ndarray
            Shape (n_q_bins,) q-values in Å⁻¹
        time_values : np.ndarray
            Shape (n_times,) time values in seconds
        """
        if q_values is None:
            q_values = np.logspace(-3, -1, n_q_bins)  # 0.001 to 0.1 Å⁻¹

        # Time values (logarithmically spaced for better correlation analysis)
        time_values = np.logspace(-6, 0, n_times)  # 1 μs to 1 s

        # Initialize intensity array
        intensity = np.zeros((n_times, n_q_bins))

        # Generate intensity for each q-bin
        for q_idx, q in enumerate(q_values):
            # Relaxation rate Γ = D*q²
            gamma = diffusion_coefficient * (q * 1e10) ** 2  # Convert Å⁻¹ to m⁻¹

            # Generate exponentially correlated intensity
            # Using Ornstein-Uhlenbeck process simulation
            dt = np.diff(time_values)
            dt = np.concatenate([[time_values[0]], dt])

            # Simulate correlated intensity
            I_mean = 1000.0  # Average intensity
            intensity_series = np.zeros(n_times)
            intensity_series[0] = I_mean

            for t_idx in range(1, n_times):
                # Ornstein-Uhlenbeck process step
                decay = np.exp(-gamma * dt[t_idx])
                intensity_series[t_idx] = (
                    decay * intensity_series[t_idx - 1]
                    + I_mean * (1 - decay)
                    + np.sqrt(2 * gamma * I_mean**2 * dt[t_idx]) * np.random.normal()
                )

            # Add noise
            noise = noise_level * I_mean * np.random.normal(size=n_times)
            intensity[:, q_idx] = np.maximum(intensity_series + noise, 0.1)

        return intensity, q_values, time_values

    def calculate_analytical_g2(
        self,
        tau_values: np.ndarray,
        q_values: np.ndarray,
        diffusion_coefficient: float = 1e-12,
        beta: float = 0.8,
    ) -> np.ndarray:
        """
        Calculate analytical g2 function for Brownian motion.

        Parameters
        ----------
        tau_values : np.ndarray
            Lag times in seconds
        q_values : np.ndarray
            Q-values in Å⁻¹
        diffusion_coefficient : float, optional
            Diffusion coefficient in m²/s, by default 1e-12
        beta : float, optional
            Coherence factor, by default 0.8

        Returns
        -------
        g2_analytical : np.ndarray
            Shape (len(tau_values), len(q_values)) analytical g2 values
        """
        g2_analytical = np.zeros((len(tau_values), len(q_values)))

        for q_idx, q in enumerate(q_values):
            gamma = diffusion_coefficient * (q * 1e10) ** 2  # Convert Å⁻¹ to m⁻¹
            g2_analytical[:, q_idx] = 1 + beta * np.exp(-2 * gamma * tau_values)

        return g2_analytical

    def create_test_hdf5_file(
        self, filepath: Path, file_type: str = "nexus", include_twotime: bool = False
    ) -> dict[str, Any]:
        """
        Create a synthetic HDF5 file with XPCS data structure.

        Parameters
        ----------
        filepath : Path
            Output file path
        file_type : str, optional
            File format type, by default "nexus"
        include_twotime : bool, optional
            Whether to include two-time correlation data, by default False

        Returns
        -------
        metadata : dict
            Dictionary containing information about the generated data
        """
        # Generate synthetic data
        intensity, q_values, time_values = self.generate_brownian_motion_intensity()

        # Calculate multi-tau correlation function
        tau_values = np.logspace(-6, 0, 100)  # 100 lag times
        g2_values = self.calculate_analytical_g2(tau_values, q_values)
        g2_errors = 0.01 * g2_values + 0.001  # Realistic error bars

        # Create HDF5 file
        with h5py.File(filepath, "w") as f:
            # Create NeXus structure
            entry = f.create_group("entry")
            entry.attrs["NX_class"] = "NXentry"

            # Instrument information
            instrument = entry.create_group("instrument")
            instrument.attrs["NX_class"] = "NXinstrument"

            detector = instrument.create_group("detector")
            detector.attrs["NX_class"] = "NXdetector"

            # Sample information
            sample = entry.create_group("sample")
            sample.attrs["NX_class"] = "NXsample"
            sample.create_dataset("temperature", data=298.15)  # Room temperature
            sample.create_dataset("description", data=b"Synthetic Brownian particles")

            # User information
            user = entry.create_group("user")
            user.create_dataset("name", data=b"XPCS Test Suite")
            user.create_dataset("email", data=b"test@xpcs-toolkit.org")

            # Exchange group for compatibility with analysis type detection
            exchange = f.create_group("exchange")
            exchange.create_dataset("tau", data=tau_values)
            exchange.create_dataset("g2", data=g2_values)
            exchange.create_dataset("norm-0-g2", data=g2_values)  # Standard XPCS format
            exchange.create_dataset("norm-0-stderr", data=g2_errors)

            # XPCS analysis results
            xpcs = f.create_group("xpcs")

            # Multi-tau correlation
            multitau = xpcs.create_group("multitau")
            multitau.create_dataset("tau", data=tau_values)
            multitau.create_dataset("g2", data=g2_values)
            multitau.create_dataset("g2_err", data=g2_errors)
            multitau.create_dataset("ql_sta", data=q_values)
            multitau.create_dataset("normalized_g2", data=g2_values)  # Standard path
            multitau.create_dataset("normalized_g2_err", data=g2_errors)
            multitau.create_dataset("delay_list", data=tau_values)
            multitau.create_dataset("ql_dyn", data=q_values)

            # Configuration
            config = multitau.create_group("config")
            config.create_dataset("analysis_type", data=b"Multitau")
            config.create_dataset("num_levels", data=8)
            config.create_dataset("num_bufs", data=16)
            config.create_dataset("stride_frame", data=1)
            config.create_dataset("avg_frame", data=1)

            # Add required timing data for get_g2_data method
            detector_1 = instrument.create_group("detector_1")
            detector_1.create_dataset("frame_time", data=1e-3)  # t0 = 1ms
            detector_1.create_dataset("count_time", data=1e-3)  # t1 = 1ms

            # Add start_time for completeness
            entry.create_dataset("start_time", data=b"2023-01-01T00:00:00Z")

            # Two-time correlation (optional)
            if include_twotime:
                twotime = xpcs.create_group("twotime")
                # Create synthetic two-time map
                c2_map = np.random.exponential(1.0, (len(tau_values), len(tau_values)))
                twotime.create_dataset("c2_map", data=c2_map)
                twotime.create_dataset("tau1", data=tau_values)
                twotime.create_dataset("tau2", data=tau_values)

        # Metadata for testing
        metadata = {
            "file_path": str(filepath),
            "file_type": file_type,
            "diffusion_coefficient": 1e-12,
            "q_values": q_values,
            "tau_values": tau_values,
            "analytical_g2": g2_values,
            "n_q_bins": len(q_values),
            "n_tau_values": len(tau_values),
            "includes_twotime": include_twotime,
            "random_seed": self.random_seed,
        }

        return metadata


def create_test_data_suite(output_dir: Path) -> dict[str, Any]:
    """
    Create a complete suite of test data files for XPCS testing.

    Parameters
    ----------
    output_dir : Path
        Directory to store test data files

    Returns
    -------
    test_suite_info : dict
        Information about all created test files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = SyntheticXPCSDataGenerator()
    test_suite_info = {"files": []}

    # Create basic multitau file
    basic_file = output_dir / "test_basic_multitau.h5"
    metadata = generator.create_test_hdf5_file(basic_file, file_type="nexus")
    test_suite_info["files"].append(metadata)

    # Create file with two-time correlation
    twotime_file = output_dir / "test_with_twotime.h5"
    metadata = generator.create_test_hdf5_file(
        twotime_file, file_type="nexus", include_twotime=True
    )
    test_suite_info["files"].append(metadata)

    # Create multiple files for batch testing
    for i in range(3):
        batch_file = output_dir / f"test_batch_{i:02d}.h5"
        generator.random_seed = 42 + i  # Different seeds for variety
        generator.__init__(generator.random_seed)
        metadata = generator.create_test_hdf5_file(batch_file, file_type="nexus")
        test_suite_info["files"].append(metadata)

    test_suite_info["total_files"] = len(test_suite_info["files"])
    test_suite_info["output_directory"] = str(output_dir)

    return test_suite_info


# Utility functions for test data access
def get_test_data_path() -> Path:
    """Get the standard test data directory path."""
    return Path(__file__).parent / "hdf5_data"


def ensure_test_data_exists() -> dict[str, Any]:
    """
    Ensure test data exists, create it if necessary.

    Returns
    -------
    test_suite_info : dict
        Information about available test data
    """
    test_data_dir = get_test_data_path()

    # Check if test data already exists
    if test_data_dir.exists() and len(list(test_data_dir.glob("*.h5"))) > 0:
        # Test data exists, return info
        existing_files = list(test_data_dir.glob("*.h5"))
        return {
            "output_directory": str(test_data_dir),
            "total_files": len(existing_files),
            "files": [{"file_path": str(f)} for f in existing_files],
            "status": "existing",
        }

    # Create new test data
    test_suite_info = create_test_data_suite(test_data_dir)
    test_suite_info["status"] = "created"
    return test_suite_info


if __name__ == "__main__":
    # Create test data for development
    test_info = ensure_test_data_exists()
    print(
        f"Test data suite: {test_info['total_files']} files in {test_info['output_directory']}"
    )
