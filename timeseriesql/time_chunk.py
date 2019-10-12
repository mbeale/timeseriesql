class TimeChunk:
    """ The TimeChuck class represents a mask over a NumpyArray """

    row_mask = None
    col_mask = None

    def __init__(self, row_mask, col_mask):
        """
        Set the intial row and column mask

        Parameters
        ----------

        row_mask : slice
            designates which rows to select
        col_mask : slice
            designates which columns to select
        """
        self.row_mask = row_mask
        self.col_mask = col_mask
