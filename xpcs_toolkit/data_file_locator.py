import os
import logging
import traceback
import datetime
import time
import warnings
from typing import Optional, List, Tuple

from .xpcs_file import XpcsDataFile
from .helper.listmodel import ListDataModel
from .fileIO.qmap_utils import QMapManager

logger = logging.getLogger(__name__)


def create_xpcs_dataset(filename: str, **kwargs) -> Optional[XpcsDataFile]:
    """
    Create an XPCS data file object from a given path.
    
    Parameters
    ----------
    filename : str
        Path to the XPCS data file.
    **kwargs
        Additional keyword arguments for XpcsDataFile initialization.
        
    Returns
    -------
    XpcsDataFile or None
        XpcsDataFile object if successful, None if failed to load.
    """
    try:
        temp = XpcsDataFile(filename, **kwargs)
    except Exception as e:
        logger.error("failed to load file: %s", filename)
        logger.error(traceback.format_exc())
        temp = None
    return temp


class DataFileLocator:
    """
    DataFileLocator manages file discovery, caching, and organization for XPCS data analysis.
    
    This class provides functionality to locate XPCS data files, maintain a cache
    of loaded data objects, and organize files for analysis workflows.
    
    Parameters
    ----------
    path : str
        Path to the directory containing XPCS data files.
    """
    
    def __init__(self, path: str):
        self.path = path
        self.source_files = ListDataModel()
        self.search_results = ListDataModel()
        self.target_files = ListDataModel()
        self.qmap_manager = QMapManager()
        self.file_cache = {}
        self.last_modified_time = None

    def set_directory_path(self, path: str) -> None:
        """Set the directory path for file location.
        
        Parameters
        ----------
        path : str
            New directory path.
        """
        self.path = path

    def clear_file_lists(self) -> None:
        """Clear the source and search result file lists."""
        self.source_files.clear()
        self.search_results.clear()

    def get_xpcs_file_list(self, rows: Optional[List[int]] = None, 
                          filter_analysis_type: Optional[str] = None, 
                          filter_fitted: bool = False) -> List[XpcsDataFile]:
        """
        Get the cached XPCS file list with optional filtering.
        
        Parameters
        ----------
        rows : list of int, optional
            List of indices to select. If None, select all files.
        filter_analysis_type : str, optional
            Filter by analysis type (e.g., 'Multitau', 'Twotime').
        filter_fitted : bool, optional
            If True, only return files with fitting results.
            
        Returns
        -------
        list of XpcsDataFile
            List of selected XPCS data file objects.
        """
        if not rows:
            selected_indices = list(range(len(self.target_files)))
        else:
            selected_indices = rows

        result_files = []
        for index in selected_indices:
            if index < 0 or index >= len(self.target_files):
                continue
            full_filename = os.path.join(self.path, self.target_files[index])
            if full_filename not in self.file_cache:
                xpcs_file_obj = create_xpcs_dataset(full_filename, qmap_manager=self.qmap_manager)
                self.file_cache[full_filename] = xpcs_file_obj
            xpcs_file_obj = self.file_cache[full_filename]
            
            if xpcs_file_obj is None:
                continue
                
            if xpcs_file_obj.fit_summary is None and filter_fitted:
                continue
            if filter_analysis_type is None:
                result_files.append(xpcs_file_obj)
            elif filter_analysis_type in xpcs_file_obj.analysis_type:
                result_files.append(xpcs_file_obj)
        return result_files

    def get_hdf_metadata(self, filename: str, filter_strings: Optional[List[str]] = None) -> dict:
        """
        Get HDF metadata information for a specific file.
        
        Parameters
        ----------
        filename : str
            Input filename (relative to the current path).
        filter_strings : list of str, optional
            List of filter strings to apply to metadata.
            
        Returns
        -------
        dict
            Dictionary containing HDF metadata information.
        """
        xpcs_file_obj = create_xpcs_dataset(
            os.path.join(self.path, filename), qmap_manager=self.qmap_manager
        )
        return xpcs_file_obj.get_hdf_metadata(filter_strings)

    def add_target_files(self, file_list: List[str], threshold: int = 256, 
                        preload: bool = True) -> None:
        """
        Add files to the target list for analysis.
        
        Parameters
        ----------
        file_list : list of str
            List of filenames to add.
        threshold : int, optional
            Maximum number of files to preload.
        preload : bool, optional
            Whether to preload file data.
        """
        if not file_list:
            return
        if preload and len(file_list) <= threshold:
            start_time = time.perf_counter()
            for filename in file_list:
                if filename in self.target_files:
                    continue
                full_filename = os.path.join(self.path, filename)
                xpcs_file_obj = create_xpcs_dataset(full_filename, qmap_manager=self.qmap_manager)
                if xpcs_file_obj is not None:
                    self.target_files.append(filename)
                    self.file_cache[full_filename] = xpcs_file_obj
            end_time = time.perf_counter()
            logger.info(f"Load {len(file_list)} files in {end_time-start_time:.3f} seconds")
        else:
            logger.info("preload disabled or too many files added")
            self.target_files.extend(file_list)
        self.last_modified_time = str(datetime.datetime.now())

    def clear_target_files(self) -> None:
        """Clear the target file list and cache."""
        self.target_files.clear()
        self.file_cache.clear()

    def remove_target_files(self, removal_list: List[str]) -> None:
        """
        Remove files from the target list.
        
        Parameters
        ----------
        removal_list : list of str
            List of filenames to remove.
        """
        for filename in removal_list:
            if filename in self.target_files:
                self.target_files.remove(filename)
            self.file_cache.pop(os.path.join(self.path, filename), None)
        if not self.target_files:
            self.clear_target_files()
        self.last_modified_time = str(datetime.datetime.now())

    def reorder_target_file(self, row: int, direction: str = "up") -> int:
        """
        Reorder a file in the target list.
        
        Parameters
        ----------
        row : int
            Row index of the file to reorder.
        direction : str, optional
            Direction to move ('up' or 'down').
            
        Returns
        -------
        int
            New position index, or -1 if no change.
        """
        size = len(self.target_files)
        assert 0 <= row < size, "check row value"
        if (direction == "up" and row == 0) or (
            direction == "down" and row == size - 1
        ):
            return -1

        item = self.target_files.pop(row)
        pos = row - 1 if direction == "up" else row + 1
        self.target_files.insert(pos, item)
        new_index = self.target_files.index(pos)
        self.last_modified_time = str(datetime.datetime.now())
        return new_index

    def search_files(self, search_value: str, filter_type: str = "prefix") -> None:
        """
        Search for files matching the given criteria.
        
        Parameters
        ----------
        search_value : str
            Search query string.
        filter_type : str, optional
            Type of filter to apply ('prefix' or 'substr').
        """
        assert filter_type in [
            "prefix",
            "substr",
        ], "filter_type must be prefix or substr"
        if filter_type == "prefix":
            selected = [x for x in self.source_files if x.startswith(search_value)]
        elif filter_type == "substr":
            filter_words = search_value.split()  # Split search query by whitespace
            selected = [x for x in self.source_files if all(word in x for word in filter_words)]
        self.search_results.replace(selected)

    def build_file_list(self, path: Optional[str] = None, 
                       file_extensions: Tuple[str, ...] = (".hdf", ".h5"), 
                       sort_method: str = "Filename") -> bool:
        """
        Build the file list from the specified directory.
        
        Parameters
        ----------
        path : str, optional
            Directory path to scan. If None, use current path.
        file_extensions : tuple of str, optional
            File extensions to include in the search.
        sort_method : str, optional
            Sorting method ('Filename', 'Time', 'Index', with optional '-reverse').
            
        Returns
        -------
        bool
            True if successful.
        """
        if path is not None:
            self.path = path
        file_list = [
            entry.name
            for entry in os.scandir(self.path)
            if entry.is_file()
            and entry.name.lower().endswith(file_extensions)
            and not entry.name.startswith(".")
        ]
        if sort_method.startswith("Filename"):
            file_list.sort()
        elif sort_method.startswith("Time"):
            file_list.sort(key=lambda x: os.path.getmtime(os.path.join(self.path, x)))
        elif sort_method.startswith("Index"):
            pass

        if sort_method.endswith("-reverse"):
            file_list.reverse()
        self.source_files.replace(file_list)
        return True

    @property
    def target(self) -> ListDataModel:
        """Alias for target_files for backward compatibility."""
        return self.target_files

    # Deprecated method aliases for backward compatibility
    def get_xf_list(self, *args, **kwargs):
        """Deprecated: Use get_xpcs_file_list() instead."""
        warnings.warn(
            "get_xf_list is deprecated, use get_xpcs_file_list instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.get_xpcs_file_list(*args, **kwargs)

    def get_hdf_info(self, fname, filter_str=None):
        """Deprecated: Use get_hdf_metadata() instead."""
        warnings.warn(
            "get_hdf_info is deprecated, use get_hdf_metadata instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.get_hdf_metadata(fname, filter_strings=filter_str)

    def set_path(self, path):
        """Deprecated: Use set_directory_path() instead."""
        warnings.warn(
            "set_path is deprecated, use set_directory_path instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.set_directory_path(path)

    def clear(self):
        """Deprecated: Use clear_file_lists() instead."""
        warnings.warn(
            "clear is deprecated, use clear_file_lists instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.clear_file_lists()

    def add_target(self, *args, **kwargs):
        """Deprecated: Use add_target_files() instead."""
        warnings.warn(
            "add_target is deprecated, use add_target_files instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.add_target_files(*args, **kwargs)

    def clear_target(self):
        """Deprecated: Use clear_target_files() instead."""
        warnings.warn(
            "clear_target is deprecated, use clear_target_files instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.clear_target_files()

    def remove_target(self, *args, **kwargs):
        """Deprecated: Use remove_target_files() instead."""
        warnings.warn(
            "remove_target is deprecated, use remove_target_files instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.remove_target_files(*args, **kwargs)

    def reorder_target(self, *args, **kwargs):
        """Deprecated: Use reorder_target_file() instead."""
        warnings.warn(
            "reorder_target is deprecated, use reorder_target_file instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.reorder_target_file(*args, **kwargs)

    def search(self, *args, **kwargs):
        """Deprecated: Use search_files() instead."""
        warnings.warn(
            "search is deprecated, use search_files instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.search_files(*args, **kwargs)

    def build(self, *args, **kwargs):
        """Deprecated: Use build_file_list() instead."""
        warnings.warn(
            "build is deprecated, use build_file_list instead",
            DeprecationWarning,
            stacklevel=2
        )
        return self.build_file_list(*args, **kwargs)


# Backward compatibility alias
class FileLocator(DataFileLocator):
    """Deprecated: Use DataFileLocator instead.
    
    This class is provided for backward compatibility only.
    New code should use DataFileLocator directly.
    """
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "FileLocator is deprecated, use DataFileLocator instead",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


if __name__ == "__main__":
    fl = DataFileLocator(path="./data")
