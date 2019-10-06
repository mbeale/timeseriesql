import unittest
import numpy as np
import os
from timeseriesql.csv_backend import CSVBackend
from timeseriesql.timeseries import TimeSeries


class TestCSVBackend(unittest.TestCase):
    def test_load_csv(self):
        data = CSVBackend(x for x in "./timeseriesql/tests/csv/basic.csv")[:]

        # created new timeseries
        self.assertTrue(isinstance(data, TimeSeries))

        # shape is expected
        self.assertEqual((11, 7), data.shape)

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
        self.assertTrue(np.array_equal(data.time[-2:], [1520640000.0, 1520726400.0]))

    def test_load_csv_filter_headers(self):
        # =, !-, not in, in

        # ==
        data = CSVBackend(x for x in "./timeseriesql/tests/csv/basic.csv" if x.label == "A")[:]
        self.assertEqual((11, 1), data.shape)
        self.assertTrue(np.array_equal(data.labels, [{"label": "A"}]))

        # !==
        data = CSVBackend(x for x in "./timeseriesql/tests/csv/basic.csv" if x.label != "B")[:]
        self.assertEqual((11, 6), data.shape)
        self.assertTrue(
            np.array_equal(
                data.labels,
                [
                    {"label": "A"},
                    {"label": "C"},
                    {"label": "D"},
                    {"label": "E"},
                    {"label": "F"},
                    {"label": "G"},
                ],
            )
        )

        # in
        data = CSVBackend(
            x for x in "./timeseriesql/tests/csv/basic.csv" if x.label in ["B", "C", "G"]
        )[:]
        self.assertEqual((11, 3), data.shape)
        self.assertTrue(
            np.array_equal(data.labels, [{"label": "B"}, {"label": "C"}, {"label": "G"}])
        )

        data = CSVBackend(
            x for x in "./timeseriesql/tests/csv/basic.csv" if x.label not in ["B", "C", "G"]
        )[:]
        self.assertEqual((11, 4), data.shape)
        self.assertTrue(
            np.array_equal(
                data.labels, [{"label": "A"}, {"label": "D"}, {"label": "E"}, {"label": "F"}]
            )
        )

    def test_load_csv_with_empty_columns(self):
        data = CSVBackend(x for x in "./timeseriesql/tests/csv/basic_with_missing_data.csv")[:]

        # created new timeseries
        self.assertTrue(isinstance(data, TimeSeries))

        # shape is expected
        self.assertEqual((11, 7), data.shape)

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

    def test_load_csv_ignore_headers(self):
        data = CSVBackend(x for x in "./timeseriesql/tests/csv/basic.csv").labels(
            [
                {"label": "one"},
                {"label": "two"},
                {"label": "three"},
                {"label": "four"},
                {"label": "five"},
                {"label": "six"},
                {"label": "seven"},
            ]
        )[:]

        # shape is expected
        self.assertEqual((11, 7), data.shape)

        # labels
        self.assertTrue(
            np.array_equal(
                data.labels,
                [
                    {"label": "one"},
                    {"label": "two"},
                    {"label": "three"},
                    {"label": "four"},
                    {"label": "five"},
                    {"label": "six"},
                    {"label": "seven"},
                ],
            )
        )

    def test_load_csv_string_date(self):
        data = CSVBackend(x for x in "./timeseriesql/tests/csv/basic_with_string_dates.csv")[:]

        tstamps = [1571296500.0 + (i * 60) for i in range(11)]
        self.assertTrue(np.array_equal(data.time, tstamps))

