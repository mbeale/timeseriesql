from typing import Optional, Union, List


class TimeChunk:
    """ The TimeChuck class represents a mask over a NumpyArray """

    row_mask: Optional[Union[int, slice]] = None
    col_mask: Optional[Union[List[int], int, slice]] = None

    def __init__(
        self,
        row_mask: Optional[Union[int, slice]],
        col_mask: Optional[Union[List[int], int, slice]],
    ):
        """
        Set the intial row and column mask

        Parameters
        ----------

        row_mask : slice, int
            designates which rows to select
        col_mask : int, list[int]
            designates which columns to select
        """
        self.row_mask = row_mask
        self.col_mask = col_mask
