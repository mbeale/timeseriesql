import numpy as np


class NumpyArray(np.ndarray):
    def __new__(self, shape, parent):
        obj = np.empty(shape, dtype=np.float64).view(self)
        obj[:] = np.nan
        obj.parent = parent
        return obj

    def __array_finalize__(self, obj):
        """Used to add labels to the object"""
        if obj is None:
            return
        self.parent = getattr(obj, "parent", None)

    def __array_wrap__(self, out_arr, context=None):
        return self.parent.wrap_new_data(out_arr)

