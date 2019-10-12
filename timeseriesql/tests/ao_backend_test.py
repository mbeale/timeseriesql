import unittest
import time
import json
from unittest import mock
from timeseriesql.backends.ao_backend import AOBackend, create_scalar_time_series
from timeseriesql.query import Plan
from timeseriesql.decompiler import Variable, FilterVariable, Constant


def mocked_requests_get(*args, **kwargs):
    class MockResponse:
        def __init__(self, json_data, status_code, response_text=None):
            self.json_data = json_data
            self.status_code = status_code
            self.text = response_text

        def json(self):
            with open('./timeseriesql/tests/json/ao_basic.json') as json_file:
                data = json.load(json_file)

            return data

    return MockResponse("ao_basic.json", 200)


class TestAOBackend(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        now = int(time.time())
        self.basic_period = {"start_time": now - 3600, "end_time": now, "resolution": 1}

    def test_execute_basic_plan(self):
        expected_plan = {
            "metrics": ["test"],
            "calc": None,
            "filters": [],
            "group": None,
            "variables": [Variable("x", labels=[])],
        }

        self.assertEqual(
            AOBackend().create_query(Plan(**expected_plan), self.basic_period),
            's("test","*",{period:"1","function":"mean"})',
        )

    def test_basic_query_with_label(self):
        expected_plan = {
            "metrics": ["test"],
            "calc": None,
            "filters": [],
            "group": None,
            "variables": [Variable("x", labels=["max"])],
        }
        self.assertEqual(
            AOBackend().create_query(Plan(**expected_plan), self.basic_period),
            's("test","*",{period:"1","function":"max"})',
        )

        expected_plan2 = {
            "metrics": ["test"],
            "calc": None,
            "filters": [],
            "group": None,
            "variables": [Variable("x", labels=["max", "mean"])],
        }
        self.assertRaises(
            AttributeError, AOBackend().create_query, Plan(**expected_plan2), self.basic_period
        )

    def test_basic_filter(self):
        expected_plan = {
            "metrics": ["test"],
            "calc": None,
            "filters": [
                {
                    "left": FilterVariable(Variable("x", labels=["label1"])),
                    "op": "==",
                    "right": FilterVariable(Constant("prod")),
                }
            ],
            "group": None,
            "variables": [Variable("x", labels=[])],
        }
        self.assertEqual(
            AOBackend().create_query(Plan(**expected_plan), self.basic_period),
            's("test",{"label1":"prod"},{period:"1","function":"mean"})',
        )
        expected_plan2 = {
            "metrics": ["test"],
            "calc": None,
            "filters": [
                {
                    "left": FilterVariable(Variable("x", labels=["label1"])),
                    "op": "in",
                    "right": FilterVariable(Constant("prod")),
                }
            ],
            "group": None,
            "variables": [Variable("x", labels=[])],
        }
        self.assertRaises(
            AttributeError, AOBackend().create_query, Plan(**expected_plan2), self.basic_period
        )

        expected_plan3 = {
            "metrics": ["test"],
            "calc": None,
            "filters": [
                {
                    "left": FilterVariable(Constant("prod")),
                    "op": "==",
                    "right": FilterVariable(Constant("prod")),
                }
            ],
            "group": None,
            "variables": [Variable("x", labels=[])],
        }
        self.assertRaises(
            AttributeError, AOBackend().create_query, Plan(**expected_plan3), self.basic_period
        )

    def test_binary_operations(self):
        expected_plan = {
            "metrics": ["test"],
            "calc": [["x", 100, "BINARY_MULTIPLY"]],
            "filters": [
                {
                    "left": FilterVariable(Variable("x", labels=["label1"])),
                    "op": "==",
                    "right": FilterVariable(Constant("prod")),
                }
            ],
            "group": None,
            "variables": [Variable("x", labels=[])],
        }
        self.assertEqual(
            AOBackend().create_query(Plan(**expected_plan), self.basic_period),
            'multiply([s("test",{"label1":"prod"},{period:"1","function":"mean"}),scale(divide([s("test",{"label1":"prod"},{period:"1","function":"mean"}),s("test",{"label1":"prod"},{period:"1","function":"mean"})]),{"factor":"100"})])',
        )

    def test_group_by(self):
        expected_plan = {
            "metrics": ["test"],
            "calc": [["x", 100, "BINARY_MULTIPLY"]],
            "filters": [
                {
                    "left": FilterVariable(Variable("x", labels=["label1"])),
                    "op": "==",
                    "right": FilterVariable(Constant("prod")),
                }
            ],
            "group": [["label1"], "mean"],
            "variables": [Variable("x", labels=[])],
        }
        expected_value = 'group_by("label1",multiply([mean(s("test",{"label1":"prod"},{period:"1","function":"mean"})),scale(divide([mean(s("test",{"label1":"prod"},{period:"1","function":"mean"})),mean(s("test",{"label1":"prod"},{period:"1","function":"mean"}))]),{"factor":"100"})]))'

        self.assertEqual(
            AOBackend().create_query(Plan(**expected_plan), self.basic_period), expected_value
        )

    def test_multiple_binary_operations(self):
        expected_plan = {
            "metrics": ["test"],
            "calc": [["x", 100, "BINARY_MULTIPLY"], ["x", "BINARY_MULTIPLY"]],
            "filters": [
                {
                    "left": FilterVariable(Variable("x", labels=["label1"])),
                    "op": "==",
                    "right": FilterVariable(Constant("prod")),
                }
            ],
            "group": None,
            "variables": [Variable("x", labels=[])],
        }
        expected_value = 'multiply([multiply([s("test",{"label1":"prod"},{period:"1","function":"mean"}),scale(divide([s("test",{"label1":"prod"},{period:"1","function":"mean"}),s("test",{"label1":"prod"},{period:"1","function":"mean"})]),{"factor":"100"})]),s("test",{"label1":"prod"},{period:"1","function":"mean"})])'

        self.assertEqual(
            AOBackend().create_query(Plan(**expected_plan), self.basic_period), expected_value
        )

        # swap order of calc values
        expected_plan2 = {
            "metrics": ["test"],
            "calc": [[100, "x", "BINARY_MULTIPLY"]],
            "filters": [
                {
                    "left": FilterVariable(Variable("x", labels=["label1"])),
                    "op": "==",
                    "right": FilterVariable(Constant("prod")),
                }
            ],
            "group": None,
            "variables": [Variable("x", labels=[])],
        }

        expected_value2 = 'multiply([scale(divide([s("test",{"label1":"prod"},{period:"1","function":"mean"}),s("test",{"label1":"prod"},{period:"1","function":"mean"})]),{"factor":"100"}), s("test",{"label1":"prod"},{period:"1","function":"mean"})])'
        self.assertEqual(
            AOBackend().create_query(Plan(**expected_plan2), self.basic_period), expected_value2
        )

        # unexpected op code
        expected_plan3 = {
            "metrics": ["test"],
            "calc": [[100, "x", "BINARY_OR"]],
            "filters": [
                {
                    "left": FilterVariable(Variable("x", labels=["label1"])),
                    "op": "==",
                    "right": FilterVariable(Constant("prod")),
                }
            ],
            "group": None,
            "variables": [Variable("x", labels=[])],
        }

        self.assertRaises(
            TypeError, AOBackend().create_query, Plan(**expected_plan3), self.basic_period
        )

    def test_scaler_create_trick(self):
        self.maxDiff = None
        series = 's("AWS.EC2.CPUUtilization",{"hostname":"ip-172-26-26-192.ec2.internal"},{period:"60","function":"mean"})'
        expected_result = 'scale(divide([s("AWS.EC2.CPUUtilization",{"hostname":"ip-172-26-26-192.ec2.internal"},{period:"60","function":"mean"}),s("AWS.EC2.CPUUtilization",{"hostname":"ip-172-26-26-192.ec2.internal"},{period:"60","function":"mean"})]),{"factor":"100"})'

        actual_result = create_scalar_time_series(series, 100)
        self.assertEqual(actual_result, expected_result)

    def test_multiple_generators(self):
        self.maxDiff = None
        expected_plan = {
            "metrics": [
                Plan(
                    **{
                        "calc": None,
                        "metrics": ["metric1"],
                        "filters": [],
                        "group": [["tag1"], "mean"],
                        "variables": [Variable("x", labels=["max"])],
                    }
                ),
                Plan(
                    **{
                        "calc": None,
                        "metrics": ["metric2"],
                        "filters": [],
                        "group": [["tag1"], "mean"],
                        "variables": [Variable("x", labels=["min"])],
                    }
                ),
            ],
            "calc": [["x", "y", "BINARY_SUBTRACT"]],
            "filters": [],
            "group": None,
            "variables": [Variable("x", labels=[]), Variable("y", labels=[])],
        }
        expected_value = 'subtract([group_by("tag1",mean(s("metric1","*",{period:"1","function":"max"}))),group_by("tag1",mean(s("metric2","*",{period:"1","function":"min"})))])'
        self.assertEqual(
            AOBackend().create_query(Plan(**expected_plan), self.basic_period), expected_value
        )

    @mock.patch("timeseriesql.backends.ao_backend.requests.get", side_effect=mocked_requests_get)
    def test_end_to_end(self, mock_requests):
        data = AOBackend(x for x in "test")[:]

        #test the labels
        self.assertEqual(data.labels, [{'service':'service1', 'name':'metric1' }])

        #measurements loaded
        self.assertEqual(len(data), 358)
        self.assertEqual(data[-1][0], 549.56976923)
