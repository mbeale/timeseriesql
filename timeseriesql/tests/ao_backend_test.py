import unittest
import time
import json
from unittest import mock
from timeseriesql.backends.ao_backend import AOBackend, create_scalar_time_series
from timeseriesql.query import Plan


def mocked_requests_get(*args, **kwargs):
    class MockResponse:
        def __init__(self, json_data, status_code, response_text=None):
            self.json_data = json_data
            self.status_code = status_code
            self.text = response_text

        def json(self):
            with open("./timeseriesql/tests/json/ao_basic.json") as json_file:
                data = json.load(json_file)

            return data

    return MockResponse("ao_basic.json", 200)


class TestAOBackend(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        now = int(time.time())
        self.basic_period = {"start_time": now - 3600, "end_time": now, "resolution": 1}

    def test_execute_basic_plan(self):
        a = AOBackend(x for x in "test")
        self.assertEqual(a.composite, 's("test","*",{period:"1","function":"mean"})')

    def test_basic_query_with_label(self):

        a = AOBackend(x.max for x in "test")
        self.assertEqual(a.composite, 's("test","*",{period:"1","function":"max"})')

    def test_basic_filter(self):
        a = AOBackend(x for x in "test" if x.label1 == "prod")
        self.assertEqual(a.composite, 's("test",{"label1":"prod"},{period:"1","function":"mean"})')

        a = AOBackend(x for x in "test" if x.label1 != "prod")
        self.assertEqual(a.composite, 's("test",{"label1":"!prod"},{period:"1","function":"mean"})')

        a = AOBackend(x for x in "test" if x.label1 in ["prod", "prod2"])
        self.assertRaises(NotImplementedError, getattr, a, "composite")

        a = AOBackend(x for x in "test" if "prod" == "prod")
        self.assertRaises(NotImplementedError, getattr, a, "composite")

    def test_binary_operations(self):
        a = AOBackend(x * 100 for x in "test" if x.label1 == "prod")
        self.assertEqual(
            a.composite,
            'multiply([s("test",{"label1":"prod"},{period:"1","function":"mean"}),scale(divide([s("test",{"label1":"prod"},{period:"1","function":"mean"}),s("test",{"label1":"prod"},{period:"1","function":"mean"})]),{"factor":"100"})])',
        )

    def test_group_by(self):
        expected_value = 'group_by("label1",multiply([mean(s("test",{"label1":"prod"},{period:"1","function":"mean"})),scale(divide([mean(s("test",{"label1":"prod"},{period:"1","function":"mean"})),mean(s("test",{"label1":"prod"},{period:"1","function":"mean"}))]),{"factor":"100"})]))'

        a = AOBackend(x * 100 for x in "test" if x.label1 == "prod").by(["label1"])
        self.assertEqual(a.composite, expected_value)

    def test_composite_func(self):

        a = AOBackend(mean(x) for x in "test")
        self.assertEqual(a.composite, 'mean(s("test","*",{period:"1","function":"mean"}))')

        a = AOBackend(bottom(x, function="min", count=10) for x in "test")
        self.assertEqual(
            a.composite,
            'bottom(s("test","*",{period:"1","function":"mean"}),{"count":"10","function":"min"})',
        )

        a = AOBackend(does_not_exist(x) for x in "test")
        self.assertRaises(NotImplementedError, getattr, a, "composite")

    def test_multiple_binary_operations(self):
        expected_value = 'multiply([multiply([s("test",{"label1":"prod"},{period:"1","function":"mean"}),scale(divide([s("test",{"label1":"prod"},{period:"1","function":"mean"}),s("test",{"label1":"prod"},{period:"1","function":"mean"})]),{"factor":"100"})]),s("test",{"label1":"prod"},{period:"1","function":"mean"})])'

        a = AOBackend(x * 100 * x for x in "test" if x.label1 == "prod")
        self.assertEqual(a.composite, expected_value)

        expected_value2 = 'multiply([scale(divide([s("test",{"label1":"prod"},{period:"1","function":"mean"}),s("test",{"label1":"prod"},{period:"1","function":"mean"})]),{"factor":"100"}), s("test",{"label1":"prod"},{period:"1","function":"mean"})])'
        a = AOBackend(100 * x for x in "test" if x.label1 == "prod")
        self.assertEqual(a.composite, expected_value2)

        a = AOBackend(x ^ 3 for x in "test")
        self.assertRaises(NotImplementedError, getattr, a, "composite")

    def test_scaler_create_trick(self):
        self.maxDiff = None
        series = 's("AWS.EC2.CPUUtilization",{"hostname":"ip-x.ec2.internal"},{period:"60","function":"mean"})'
        expected_result = 'scale(divide([s("AWS.EC2.CPUUtilization",{"hostname":"ip-x.ec2.internal"},{period:"60","function":"mean"}),s("AWS.EC2.CPUUtilization",{"hostname":"ip-x.ec2.internal"},{period:"60","function":"mean"})]),{"factor":"100"})'

        actual_result = create_scalar_time_series(series, 100)
        self.assertEqual(actual_result, expected_result)

    def test_multiple_generators(self):
        self.maxDiff = None
        expected_value = 'group_by("tag1",subtract([mean(s("metric1","*",{period:"1","function":"max"})),mean(s("metric2","*",{period:"1","function":"min"}))]))'
        a = AOBackend(
            x - y for x, y in AOBackend((x.max for x in "metric1"), (x.min for x in "metric2"))
        ).by("tag1")
        self.assertEqual(a.composite, expected_value)

    def test_multiple_functions(self):
        expected_value = 'sum(derive(s("test","*",{period:"1","function":"mean"})))'
        a = AOBackend(sum(derive(x)) for x in "test")
        self.assertEqual(a.composite, expected_value)

    def test_complex_composite_1(self):
        import dis

        expected_result = 'multiply([divide([zero_fill(sum(s("metric1","*",{period:"1","function":"sum"}))),zero_fill(sum(s("metric2","*",{period:"1","function":"sum"})))]),scale(divide([divide([zero_fill(sum(s("metric1","*",{period:"1","function":"sum"}))),zero_fill(sum(s("metric2","*",{period:"1","function":"sum"})))]),divide([zero_fill(sum(s("metric1","*",{period:"1","function":"sum"}))),zero_fill(sum(s("metric2","*",{period:"1","function":"sum"})))])]),{"factor":"100"})])'
        gen = (
            zero_fill(sum(x)) / zero_fill(sum(y)) * 100
            for x, y in AOBackend((x.sum for x in "metric1"), (x.sum for x in "metric2"))
        )
        a = AOBackend(gen)
        self.assertEqual(a.composite, expected_result)

    @mock.patch("timeseriesql.backends.ao_backend.requests.get", side_effect=mocked_requests_get)
    def test_end_to_end(self, mock_requests):
        data = AOBackend(x for x in "test")[:]

        # test the labels
        self.assertEqual(data.labels, [{"service": "service1", "name": "metric1"}])

        # measurements loaded
        self.assertEqual(len(data), 358)
        self.assertEqual(data[-1][0], 549.56976923)

    @mock.patch("timeseriesql.backends.ao_backend.requests.get", side_effect=mocked_requests_get)
    def test_raw_composite(self, mock_requests):
        composite = 's("some_metric", "*")'

        a = AOBackend(composite)
        self.assertEqual(a.composite, composite)

        data = AOBackend(composite)[:]
        # measurements loaded
        self.assertEqual(len(data), 358)
        self.assertEqual(data[-1][0], 549.56976923)

