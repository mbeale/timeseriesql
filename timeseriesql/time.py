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
            vfunc = np.vectorize(lambda x: np.datetime64(int(x), "s"))
            self._dt = vfunc(self)
        return self._dt
