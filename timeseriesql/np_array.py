import numpy as np


class NumpyArray(np.ndarray):
    """
    This class extends numpy.array to be used with the TimeSeries class

    """

    def __new__(self, shape, parent):
        """
        Create a new array

        Parameters
        ----------
        shape : tuple
            the size to allocate for the array
        parent : TimeSeries
            the TimeSeries object that owns this array
        """
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
        """ Wrap the array as a TimeSeries object after processing """
        return self.parent.wrap_new_data(out_arr)

