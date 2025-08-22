import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import os
import logging
import warnings

from .data_file_locator import DataFileLocator
from .module import saxs2d, saxs1d, intt, stability, g2mod, tauq, twotime
from .module.average_toolbox import AverageToolbox
from .helper.listmodel import TableDataModel
from .mpl_compat import DataTreeWidget
from .xpcs_file import XpcsDataFile

logger = logging.getLogger(__name__)


class AnalysisKernel(DataFileLocator):
    """
    AnalysisKernel provides comprehensive analysis functionality for XPCS data.
    
    This class serves as the main analysis engine, coordinating various analysis
    modules and managing data processing workflows for X-ray Photon Correlation
    Spectroscopy experiments.
    
    Parameters
    ----------
    path : str
        Path to the directory containing XPCS data files.
    statusbar : object, optional
        Status bar widget for progress updates.
    """
    
    def __init__(self, path: str, statusbar=None):
        super().__init__(path)
        self.statusbar = statusbar
        self.metadata = None
        self.reset_metadata()
        self.path = path
        self.average_toolbox = AverageToolbox(path)
        self.average_worker = TableDataModel()
        self.average_job_id = 0
        self.average_worker_active = {}
        self.current_dataset = None

    def reset_metadata(self) -> None:
        """Reset analysis metadata to default values."""
        self.metadata = {
            # saxs 1d:
            "saxs_1d_background_filename": None,
            "saxs_1d_background_file": None,
            # averaging
            "average_file_list": None,
            "average_intensity_minmax": None,
            "average_g2_data": None,
        }

    def reset_kernel(self) -> None:
        """Reset the entire analysis kernel to initial state."""
        self.clear_target()
        self.reset_metadata()

    def select_background_file(self, filename: str) -> None:
        """Select a background file for SAXS 1D analysis.
        
        Parameters
        ----------
        filename : str
            Path to the background file.
        """
        base_filename = os.path.basename(filename)
        self.metadata["saxs_1d_background_filename"] = base_filename
        self.metadata["saxs_1d_background_file"] = XpcsDataFile(filename)

    def get_data_tree(self, rows: List[int]) -> Optional[object]:
        """Get data tree widget for selected files.
        
        Parameters
        ----------
        rows : list of int
            Row indices of selected files.
            
        Returns
        -------
        object or None
            Data tree widget or None if no files selected.
        """
        xpcs_file_list = self.get_xpcs_file_list(rows)
        if xpcs_file_list:
            return xpcs_file_list[0].get_pg_tree()
        else:
            return None

    def get_fitting_tree(self, rows: List[int]) -> object:
        """Get fitting results tree widget.
        
        Parameters
        ----------
        rows : list of int
            Row indices of selected files.
            
        Returns
        -------
        object
            Fitting results tree widget.
        """
        xpcs_file_list = self.get_xpcs_file_list(rows, filter_analysis_type="Multitau")
        result = {}
        for xpcs_file in xpcs_file_list:
            result[xpcs_file.label] = xpcs_file.get_fitting_info(mode="g2_fitting")
        tree = DataTreeWidget(data=result)
        tree.setWindowTitle("fitting summary")
        tree.resize(1024, 800)
        return tree

    def plot_g2_function(self, handler, q_range: Tuple[float, float], 
                        time_range: Tuple[float, float], y_range: Tuple[float, float],
                        rows: Optional[List[int]] = None, **kwargs) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Plot G2 correlation function.
        
        Parameters
        ----------
        handler : object
            Plot handler object.
        q_range : tuple of float
            (q_min, q_max) range for q-values.
        time_range : tuple of float
            (t_min, t_max) range for time values.
        y_range : tuple of float
            (y_min, y_max) range for y-values.
        rows : list of int, optional
            Row indices of selected files.
        **kwargs
            Additional plotting parameters.
            
        Returns
        -------
        tuple
            (q_values, time_elapsed) or (None, None) if no data.
        """
        xpcs_file_list = self.get_xpcs_file_list(rows=rows, filter_analysis_type="Multitau")
        if xpcs_file_list:
            g2mod.pg_plot(
                handler, xpcs_file_list, q_range, time_range, y_range, rows=rows, **kwargs
            )
            q, time_elapsed, *unused = g2mod.get_data(xpcs_file_list)
            return q, time_elapsed
        else:
            return None, None

    def plot_q_space_map(self, handler, rows: Optional[List[int]] = None, 
                        target: Optional[str] = None) -> None:
        """Plot q-space map visualization.
        
        Parameters
        ----------
        handler : object
            Plot handler object.
        rows : list of int, optional
            Row indices of selected files.
        target : str, optional
            Type of map to plot ('scattering', 'dynamic_roi_map', 'static_roi_map').
        """
        xpcs_file_list = self.get_xpcs_file_list(rows=rows)
        if xpcs_file_list:
            if target == "scattering":
                value = np.log10(xpcs_file_list[0].saxs_2d + 1)
                vmin, vmax = np.percentile(value, (2, 98))
                handler.setImage(value, levels=(vmin, vmax))
            elif target == "dynamic_roi_map":
                handler.setImage(xpcs_file_list[0].dqmap)
            elif target == "static_roi_map":
                handler.setImage(xpcs_file_list[0].sqmap)

    def plot_tau_vs_q_preview(self, handler=None, rows: Optional[List[int]] = None) -> None:
        """Plot tau vs q preview.
        
        Parameters
        ----------
        handler : object, optional
            Plot handler object.
        rows : list of int, optional
            Row indices of selected files.
        """
        xpcs_file_list = self.get_xpcs_file_list(rows=rows, filter_analysis_type="Multitau")
        filtered_list = [xf for xf in xpcs_file_list if xf.fit_summary is not None]
        tauq.plot_pre(filtered_list, handler)

    def plot_tau_vs_q(
        self,
        handler=None,
        bounds=None,
        rows: Optional[List[int]] = None,
        plot_type: int = 3,
        fit_flag=None,
        offset=None,
        q_range: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, str]:
        """Plot tau vs q analysis with fitting.
        
        Parameters
        ----------
        handler : object, optional
            Plot handler object.
        bounds : tuple, optional
            Fitting bounds.
        rows : list of int, optional
            Row indices of selected files.
        plot_type : int, optional
            Type of plot to generate.
        fit_flag : array-like, optional
            Fitting flags.
        offset : float, optional
            Y-axis offset for multiple curves.
        q_range : tuple of float, optional
            (q_min, q_max) range for fitting.
            
        Returns
        -------
        dict
            Fitting results for each file.
        """
        xpcs_file_list = self.get_xpcs_file_list(
            rows=rows, filter_analysis_type="Multitau", filter_fitted=True
        )
        result = {}
        for xpcs_file in xpcs_file_list:
            if xpcs_file.fit_summary is None:
                logger.info("g2 fitting is not available for %s", xpcs_file.filename)
            else:
                xpcs_file.fit_tauq(q_range, bounds, fit_flag)
                result[xpcs_file.label] = xpcs_file.get_fitting_info(mode="tauq_fitting")

        if len(result) > 0:
            tauq.plot(
                xpcs_file_list, hdl=handler, q_range=q_range, offset=offset, plot_type=plot_type
            )

        return result

    def get_info_at_mouse_position(self, rows: List[int], x: int, y: int) -> Optional[str]:
        """Get information at mouse position on detector.
        
        Parameters
        ----------
        rows : list of int
            Row indices of selected files.
        x, y : int
            Pixel coordinates.
            
        Returns
        -------
        str or None
            Information string at the position.
        """
        xpcs_file = self.get_xpcs_file_list(rows)
        if xpcs_file:
            info = xpcs_file[0].get_info_at_position(x, y)
            return info

    def plot_saxs_2d(self, *args, rows: Optional[List[int]] = None, **kwargs) -> None:
        """Plot 2D SAXS pattern.
        
        Parameters
        ----------
        *args
            Positional arguments for plotting.
        rows : list of int, optional
            Row indices of selected files.
        **kwargs
            Keyword arguments for plotting.
        """
        xpcs_file_list = self.get_xpcs_file_list(rows)[0:1]
        if xpcs_file_list:
            saxs2d.plot(xpcs_file_list[0], *args, **kwargs)

    def add_region_of_interest(self, handler, **kwargs) -> None:
        """Add region of interest to analysis.
        
        Parameters
        ----------
        handler : object
            ROI handler object.
        **kwargs
            ROI parameters.
        """
        xpcs_file_list = self.get_xpcs_file_list()
        center = (xpcs_file_list[0].bcx, xpcs_file_list[0].bcy)
        if kwargs["sl_type"] == "Pie":
            handler.add_roi(cen=center, radius=100, **kwargs)
        elif kwargs["sl_type"] == "Circle":
            radius_vertical = min(xpcs_file_list[0].mask.shape[0] - center[1], center[1])
            radius_horizontal = min(xpcs_file_list[0].mask.shape[1] - center[0], center[0])
            radius = min(radius_horizontal, radius_vertical) * 0.8

            handler.add_roi(cen=center, radius=radius, label="RingA", **kwargs)
            handler.add_roi(cen=center, radius=0.8 * radius, label="RingB", **kwargs)

    def plot_saxs_1d(self, pg_handler, mp_handler, **kwargs) -> None:
        """Plot 1D SAXS data.
        
        Parameters
        ----------
        pg_handler : object
            PyQtGraph handler.
        mp_handler : object
            Matplotlib handler.
        **kwargs
            Plotting parameters.
        """
        xpcs_file_list = self.get_xpcs_file_list()
        if xpcs_file_list:
            roi_list = pg_handler.get_roi_list()
            saxs1d.pg_plot(
                xpcs_file_list,
                mp_handler,
                bkg_file=self.metadata["saxs_1d_background_file"],
                roi_list=roi_list,
                **kwargs
            )

    def export_saxs_1d_data(self, pg_handler, folder: str) -> None:
        """Export 1D SAXS data to files.
        
        Parameters
        ----------
        pg_handler : object
            PyQtGraph handler containing ROI information.
        folder : str
            Output folder path.
        """
        xpcs_file_list = self.get_xpcs_file_list()
        roi_list = pg_handler.get_roi_list()
        for xpcs_file in xpcs_file_list:
            xpcs_file.export_saxs1d(roi_list, folder)

    def switch_saxs_1d_line(self, mp_handler, line_builder_type: str) -> None:
        """Switch SAXS 1D line builder type.
        
        Parameters
        ----------
        mp_handler : object
            Matplotlib handler.
        line_builder_type : str
            Type of line builder to use.
        """
        pass
        # saxs1d.switch_line_builder(mp_handler, line_builder_type)

    def plot_two_time_correlation(self, handler, rows: Optional[List[int]] = None, 
                                 **kwargs) -> Optional[List[str]]:
        """Plot two-time correlation function.
        
        Parameters
        ----------
        handler : object
            Plot handler object.
        rows : list of int, optional
            Row indices of selected files.
        **kwargs
            Plotting parameters.
            
        Returns
        -------
        list of str or None
            Q-bin labels if new dataset, None otherwise.
        """
        xpcs_file_list = self.get_xpcs_file_list(rows, filter_analysis_type="Twotime")
        if len(xpcs_file_list) == 0:
            return None
        xpcs_file = xpcs_file_list[0]
        new_qbin_labels = None
        if self.current_dataset is None or self.current_dataset.filename != xpcs_file.filename:
            self.current_dataset = xpcs_file
            new_qbin_labels = xpcs_file.get_twotime_qbin_labels()
        twotime.plot_twotime(xpcs_file, handler, **kwargs)
        return new_qbin_labels

    def plot_intensity_vs_time(self, pg_handler, rows: Optional[List[int]] = None, **kwargs) -> None:
        """Plot intensity vs time.
        
        Parameters
        ----------
        pg_handler : object
            PyQtGraph handler.
        rows : list of int, optional
            Row indices of selected files.
        **kwargs
            Plotting parameters.
        """
        xpcs_file_list = self.get_xpcs_file_list(rows=rows)
        intt.plot(xpcs_file_list, pg_handler, **kwargs)

    def plot_stability_analysis(self, mp_handler, rows: Optional[List[int]] = None, **kwargs) -> None:
        """Plot stability analysis.
        
        Parameters
        ----------
        mp_handler : object
            Matplotlib handler.
        rows : list of int, optional
            Row indices of selected files.
        **kwargs
            Plotting parameters.
        """
        xpcs_file_obj = self.get_xpcs_file_list(rows)[0]
        stability.plot(xpcs_file_obj, mp_handler, **kwargs)

    def submit_averaging_job(self, *args, **kwargs) -> None:
        """Submit a data averaging job.
        
        Parameters
        ----------
        *args
            Job setup arguments.
        **kwargs
            Job setup keyword arguments.
        """
        if len(self.target) <= 0:
            logger.error("no average target is selected")
            return
        worker = AverageToolbox(self.path, flist=self.target, jid=self.average_job_id)
        worker.setup(*args, **kwargs)
        self.average_worker.append(worker)
        logger.info("create average job, ID = %s", worker.jid)
        self.average_job_id += 1
        self.target.clear()

    def remove_averaging_job(self, index: int) -> None:
        """Remove an averaging job.
        
        Parameters
        ----------
        index : int
            Index of the job to remove.
        """
        self.average_worker.pop(index)

    def update_averaging_info(self, job_id: int) -> None:
        """Update averaging job information.
        
        Parameters
        ----------
        job_id : int
            ID of the job to update.
        """
        self.average_worker.layoutChanged.emit()
        if 0 <= job_id < len(self.average_worker):
            self.average_worker[job_id].update_plot()

    def update_averaging_values(self, data: Tuple[Any, float]) -> None:
        """Update averaging values.
        
        Parameters
        ----------
        data : tuple
            (key, value) pair for updating.
        """
        key, val = data[0], data[1]
        if self.average_worker_active[key] is None:
            self.average_worker_active[key] = [0, np.zeros(128, dtype=np.float32)]
        record = self.average_worker_active[key]
        if record[0] == record[1].size:
            new_g2 = np.zeros(record[1].size * 2, dtype=np.float32)
            new_g2[0 : record[0]] = record[1]
            record[1] = new_g2
        record[1][record[0]] = val
        record[0] += 1

    def export_g2_data(self) -> None:
        """Export G2 correlation data."""
        pass

    # Deprecated method aliases for backward compatibility
    def get_xf_list(self, *args, **kwargs):
        """Deprecated: Use get_xpcs_file_list() instead."""
        warnings.warn(
            "get_xf_list is deprecated, use get_xpcs_file_list instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.get_xpcs_file_list(*args, **kwargs)

    def reset_meta(self):
        """Deprecated: Use reset_metadata() instead."""
        warnings.warn(
            "reset_meta is deprecated, use reset_metadata instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.reset_metadata()

    def select_bkgfile(self, fname):
        """Deprecated: Use select_background_file() instead."""
        warnings.warn(
            "select_bkgfile is deprecated, use select_background_file instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.select_background_file(fname)

    def get_pg_tree(self, rows):
        """Deprecated: Use get_data_tree() instead."""
        warnings.warn(
            "get_pg_tree is deprecated, use get_data_tree instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.get_data_tree(rows)

    def plot_g2(self, *args, **kwargs):
        """Deprecated: Use plot_g2_function() instead."""
        warnings.warn(
            "plot_g2 is deprecated, use plot_g2_function instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.plot_g2_function(*args, **kwargs)

    def plot_qmap(self, *args, **kwargs):
        """Deprecated: Use plot_q_space_map() instead."""
        warnings.warn(
            "plot_qmap is deprecated, use plot_q_space_map instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.plot_q_space_map(*args, **kwargs)

    def plot_tauq_pre(self, *args, **kwargs):
        """Deprecated: Use plot_tau_vs_q_preview() instead."""
        warnings.warn(
            "plot_tauq_pre is deprecated, use plot_tau_vs_q_preview instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.plot_tau_vs_q_preview(*args, **kwargs)

    def plot_tauq(self, *args, **kwargs):
        """Deprecated: Use plot_tau_vs_q() instead."""
        warnings.warn(
            "plot_tauq is deprecated, use plot_tau_vs_q instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.plot_tau_vs_q(*args, **kwargs)

    def get_info_at_mouse(self, *args, **kwargs):
        """Deprecated: Use get_info_at_mouse_position() instead."""
        warnings.warn(
            "get_info_at_mouse is deprecated, use get_info_at_mouse_position instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.get_info_at_mouse_position(*args, **kwargs)

    def add_roi(self, *args, **kwargs):
        """Deprecated: Use add_region_of_interest() instead."""
        warnings.warn(
            "add_roi is deprecated, use add_region_of_interest instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.add_region_of_interest(*args, **kwargs)

    def export_saxs_1d(self, *args, **kwargs):
        """Deprecated: Use export_saxs_1d_data() instead."""
        warnings.warn(
            "export_saxs_1d is deprecated, use export_saxs_1d_data instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.export_saxs_1d_data(*args, **kwargs)

    def switch_saxs1d_line(self, *args, **kwargs):
        """Deprecated: Use switch_saxs_1d_line() instead."""
        warnings.warn(
            "switch_saxs1d_line is deprecated, use switch_saxs_1d_line instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.switch_saxs_1d_line(*args, **kwargs)

    def plot_twotime(self, *args, **kwargs):
        """Deprecated: Use plot_two_time_correlation() instead."""
        warnings.warn(
            "plot_twotime is deprecated, use plot_two_time_correlation instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.plot_two_time_correlation(*args, **kwargs)

    def plot_intt(self, *args, **kwargs):
        """Deprecated: Use plot_intensity_vs_time() instead."""
        warnings.warn(
            "plot_intt is deprecated, use plot_intensity_vs_time instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.plot_intensity_vs_time(*args, **kwargs)

    def plot_stability(self, *args, **kwargs):
        """Deprecated: Use plot_stability_analysis() instead."""
        warnings.warn(
            "plot_stability is deprecated, use plot_stability_analysis instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.plot_stability_analysis(*args, **kwargs)

    def submit_job(self, *args, **kwargs):
        """Deprecated: Use submit_averaging_job() instead."""
        warnings.warn(
            "submit_job is deprecated, use submit_averaging_job instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.submit_averaging_job(*args, **kwargs)

    def remove_job(self, *args, **kwargs):
        """Deprecated: Use remove_averaging_job() instead."""
        warnings.warn(
            "remove_job is deprecated, use remove_averaging_job instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.remove_averaging_job(*args, **kwargs)

    def update_avg_info(self, *args, **kwargs):
        """Deprecated: Use update_averaging_info() instead."""
        warnings.warn(
            "update_avg_info is deprecated, use update_averaging_info instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.update_averaging_info(*args, **kwargs)

    def update_avg_values(self, *args, **kwargs):
        """Deprecated: Use update_averaging_values() instead."""
        warnings.warn(
            "update_avg_values is deprecated, use update_averaging_values instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.update_averaging_values(*args, **kwargs)

    def export_g2(self):
        """Deprecated: Use export_g2_data() instead."""
        warnings.warn(
            "export_g2 is deprecated, use export_g2_data instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.export_g2_data()


# Backward compatibility alias
class ViewerKernel(AnalysisKernel):
    """Deprecated: Use AnalysisKernel instead.
    
    This class is provided for backward compatibility only.
    New code should use AnalysisKernel directly.
    """
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ViewerKernel is deprecated, use AnalysisKernel instead",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    flist = os.listdir("./data")
    dv = AnalysisKernel("./data")
