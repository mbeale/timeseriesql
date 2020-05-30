from __future__ import annotations
import numpy as np  # type: ignore
from typing import Tuple, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .time_chunk import TimeChunk
    from .timeseries import TimeSeries


class NumpyArray(np.ndarray):
    """
    This class extends numpy.array to be used with the TimeSeries class

    """

    parent: "TimeSeries"

    def __new__(self, shape: Tuple[int, Optional[int]], parent: "TimeSeries"):
        """
        Create a new array

        Parameters
        ----------
        shape : tuple
            the size to allocate for the array
        parent : TimeSeries
            the TimeSeries object that owns this array
        """
        obj: np.ndarray = np.empty(shape, dtype=np.float64).view(self)
        obj[:] = np.nan
        obj.parent = parent
        return obj

    def __array_finalize__(self, obj):
        """Used to add labels to the object"""
        if obj is None:
            return
        self.parent = getattr(obj, "parent", None)

    def __array_wrap__(self, out_arr: NumpyArray, context=None) -> "TimeSeries":
        """ Wrap the array as a TimeSeries object after processing """
        return self.parent.wrap_new_data(out_arr)
