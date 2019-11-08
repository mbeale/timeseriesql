import numpy as np
import re

second_conversions = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800, "y": 31536000}


def convert_string_to_seconds(s):
    try:
        _, n, uom = re.split("(\d+)", s)
        n = int(n)
        if uom not in second_conversions.keys():
            raise ValueError(
                f"{uom} is not valid in the list of valid values: {second_conversions.keys()}"
            )
        return n * second_conversions[uom]
    except:
        raise ValueError(f"Invalid format.  Expecting NumberUOM (15h) but instead recieved {s}")


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
