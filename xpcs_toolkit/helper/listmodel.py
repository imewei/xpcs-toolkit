import logging

logger = logging.getLogger(__name__)


class MockSignal:
    """Mock Qt signal for compatibility"""
    def emit(self, *args, **kwargs):
        # No-op for headless operation
        pass


class ListDataModel:
    """Plain Python list wrapper that mimics Qt model interface"""
    def __init__(self, input_list=None, max_display=16384) -> None:
        if input_list is None:
            self.input_list = []
        else:
            self.input_list = input_list
        self.max_display = max_display
        self.layoutChanged = MockSignal()

    def data(self, index, role=None):
        """Compatibility method - returns string representation"""
        if hasattr(index, 'row'):  # Qt model index
            row = index.row()
        else:  # Plain integer
            row = index
        
        if 0 <= row < len(self.input_list):
            return str(self.input_list[row])
        return None

    def rowCount(self, index=None):
        """Return number of rows (items)"""
        return min(self.max_display, len(self.input_list))

    def extend(self, new_input_list):
        self.input_list.extend(new_input_list)
        self.layoutChanged.emit()

    def append(self, new_item):
        self.input_list.append(new_item)
        self.layoutChanged.emit()

    def replace(self, new_input_list):
        self.input_list.clear()
        self.extend(new_input_list)

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, i):
        return self.input_list[i]

    def pop(self, i=-1):
        return self.input_list.pop(i)
    
    def insert(self, i, item):
        self.input_list.insert(i, item)
        self.layoutChanged.emit()

    def copy(self):
        return self.input_list.copy()
        self.layoutChanged.emit()

    def remove(self, x):
        self.input_list.remove(x)
        self.layoutChanged.emit()

    def clear(self):
        self.input_list.clear()
        self.layoutChanged.emit()


class TableDataModel:
    """Plain Python table model that mimics Qt table model interface"""
    def __init__(self, input_list=None, max_display=16384) -> None:
        if input_list is None:
            self.input_list = []
        else:
            self.input_list = input_list
        self.max_display = max_display
        self.xlabels = ['id', 'size', 'progress', 'start', 'ETA (s)',
                        'finish', 'fname']
        self.layoutChanged = MockSignal()

    def data(self, index, role=None):
        """Get data at specific index"""
        if hasattr(index, 'row'):  # Qt model index
            row, col = index.row(), index.column()
        elif isinstance(index, tuple):  # (row, col) tuple
            row, col = index
        else:  # Single index for row
            row, col = index, 0
            
        if 0 <= row < len(self.input_list) and 0 <= col < len(self.xlabels):
            x = self.input_list[row]
            ret = [x.jid, x.size, x._progress, x.stime, x.eta, x.etime,
                   x.short_name]
            return ret[col]
        return None

    def rowCount(self, index=None):
        """Return number of rows"""
        return min(self.max_display, len(self.input_list))

    def columnCount(self, index=None):
        """Return number of columns"""
        return len(self.xlabels)

    def headerData(self, section, orientation=None, role=None):
        """Return header data for columns"""
        if 0 <= section < len(self.xlabels):
            return self.xlabels[section]
        return None

    def extend(self, new_input_list):
        self.input_list.extend(new_input_list)
        self.layoutChanged.emit()

    def append(self, new_item):
        self.input_list.append(new_item)
        self.layoutChanged.emit()

    def replace(self, new_input_list):
        self.input_list.clear()
        self.extend(new_input_list)

    def pop(self, index):
        if 0 <= index < self.__len__():
            self.input_list.pop(index)
            self.layoutChanged.emit()

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, i):
        return self.input_list[i]

    def copy(self):
        return self.input_list.copy()

    def remove(self, x):
        self.input_list.remove(x)

    def clear(self):
        self.input_list.clear()


def test():
    a = ['a', 'b', 'c']
    model = ListDataModel(a)
    for n in range(len(model)):
        print(model[n])


if __name__ == "__main__":
    test()
