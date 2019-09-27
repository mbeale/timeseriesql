class TimeChunk:

    row_mask = None
    col_mask = None
    axis = 0
    collapse_index = True

    def __init__(self, row_mask, col_mask, axis=1, collapse_index=True):
        self.row_mask = row_mask
        self.col_mask = col_mask
        self.axis = axis
        self.collapse_index = collapse_index
