import numpy as np


class TimeIndex(np.ndarray):
    """ 
    TimeIndex
    ----------

    This class converts a a float array to a np.datetime64 array
    """

    _dt = None

    def __new__(self, data):
        """Creates a new array based on the based on a float array"""
        obj = np.array(data, dtype=np.float).view(self)
        obj._dt = None
        return obj

    def __array_finalize__(self, obj):
        """ Finalize the array """
        if obj is None:
            return

    @property
    def dt(self):
        """ creates a datetime object for the time index """
        if not self._dt:
            self._dt = np.array(
                [np.datetime64(int(t), "s") for t in np.nditer(self)], dtype=np.datetime64
            )
        return self._dt
