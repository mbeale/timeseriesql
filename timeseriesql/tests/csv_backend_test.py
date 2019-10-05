import unittest
import numpy as np
import os
from timeseriesql.csv_backend import CSVBackend
from timeseriesql.timeseries import TimeSeries


class TestCSVBackend(unittest.TestCase):
    def test_load_csv(self):
        os.path.join(os.path.dirname(__file__))
        data = CSVBackend(x for x in "csv/basic.csv")[:]

        # created new timeseries
        self.assertTrue(isinstance(data, TimeSeries))

        # shape is expected
        self.assertEqual((11, 7), t.shape)

        # labels
        self.assertTrue(
            np.array_equal(
                data.labels,
                [
                    {"label": "A"},
                    {"label": "B"},
                    {"label": "C"},
                    {"label": "D"},
                    {"label": "E"},
                    {"label": "F"},
                    {"label": "G"},
                ],
            )
        )

        # time index is correct
        self.assertTrue(np.array_equal(data.time[-2], [1520640000.0, 1520726400.0]))

    def test_load_csv_filter_headers(self):
        # =, !-, not in, in
        pass

    def test_load_csv_with_empty_columns(self):
        pass

    def test_load_csv_ignore_headers(self):
        pass

    def test_load_csv_string_date(self):
        pass
