class TimeChunk:

    row_mask = None
    col_mask = None

    def __init__(self, row_mask, col_mask):
        self.row_mask = row_mask
        self.col_mask = col_mask
