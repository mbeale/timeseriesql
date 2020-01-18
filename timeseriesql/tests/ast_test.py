import unittest
from timeseriesql.ast import (
    AST,
    Value,
    Filter,
    LoadAttr,
    CompareEqual,
    BinaryMultiply,
    BinarySubtract,
    BinaryFloorDivide,
    BinaryTrueDivide,
    BinaryMatrixMultiply,
    BinaryPower,
    BinaryModulo,
    BinaryAdd,
    FuncCall,
    FuncArgs,
    Metric,
    Group,
)
from timeseriesql.query import Query


class TestAST(unittest.TestCase):
    def test_basic_query(self):
        a = AST((x for x in "test"))
        actual = a.decompile()
        expected = Metric("test")
        self.assertEqual(actual, expected)

    def test_basic_query_with_label(self):
        a = AST((x.mean for x in "test"))
        actual = a.decompile()
        expected = LoadAttr(Metric("test"), Value("mean"))
        self.assertEqual(actual, expected)

    def test_basic_filter(self):
        self.maxDiff = None
        a = AST((x for x in "test" if x.label1 == "prod"))
        actual = a.decompile()
        expected = Filter(Metric("test"), CompareEqual(Value("label1"), Value("prod")))
        self.assertEqual(actual, expected)

    def test_binary_operations(self):
        a = AST((x * 100 for x in "test" if x.label1 == "prod"))
        actual = a.decompile()
        expected = BinaryMultiply(
            Filter(Metric("test"), CompareEqual(Value("label1"), Value("prod"))), Value(100)
        )
        self.assertEqual(actual, expected)
        a = AST((x / 100 for x in "test" if x.label1 == "prod"))
        actual = a.decompile()
        expected = BinaryTrueDivide(
            Filter(Metric("test"), CompareEqual(Value("label1"), Value("prod"))), Value(100)
        )
        self.assertEqual(actual, expected)
        a = AST((x // 100 for x in "test" if x.label1 == "prod"))
        actual = a.decompile()
        expected = BinaryFloorDivide(
            Filter(Metric("test"), CompareEqual(Value("label1"), Value("prod"))), Value(100)
        )
        self.assertEqual(actual, expected)
        a = AST((x @ 100 for x in "test" if x.label1 == "prod"))
        actual = a.decompile()
        expected = BinaryMatrixMultiply(
            Filter(Metric("test"), CompareEqual(Value("label1"), Value("prod"))), Value(100)
        )
        self.assertEqual(actual, expected)
        a = AST((x % 100 for x in "test" if x.label1 == "prod"))
        actual = a.decompile()
        expected = BinaryModulo(
            Filter(Metric("test"), CompareEqual(Value("label1"), Value("prod"))), Value(100)
        )
        self.assertEqual(actual, expected)
        a = AST((x ** 2 for x in "test" if x.label1 == "prod"))
        actual = a.decompile()
        expected = BinaryPower(
            Filter(Metric("test"), CompareEqual(Value("label1"), Value("prod"))), Value(2)
        )
        self.assertEqual(actual, expected)

    def test_group_by(self):
        a = AST((x * 100 for x in "test" if x.label1 == "prod"), ["label1"])
        actual = a.decompile()
        expected = Group(
            BinaryMultiply(
                Filter(Metric("test"), CompareEqual(Value("label1"), Value("prod"))), Value(100)
            ),
            Value(["label1"]),
        )
        self.assertEqual(actual, expected)

    def test_multiple_binary_operations(self):
        a = AST((x * 100 * x for x in "test" if x.label1 == "prod"))
        actual = a.decompile()
        expected = BinaryMultiply(
            BinaryMultiply(
                Filter(Metric("test"), CompareEqual(Value("label1"), Value("prod"))), Value(100)
            ),
            Filter(Metric("test"), CompareEqual(Value("label1"), Value("prod"))),
        )
        self.assertEqual(actual, expected)

    def test_multiple_generators(self):
        self.maxDiff = None
        q = Query(
            x - y
            for x, y in Query((x.max for x in "metric1"), (x.min for x in "metric2")).by("tag1")
        )
        a = AST(q.generators, q.groupings)
        actual = a.decompile()
        expected = Group(
            BinarySubtract(
                LoadAttr(Metric("metric1"), Value("max")), LoadAttr(Metric("metric2"), Value("min"))
            ),
            Value([["tag1"], "mean"]),
        )
        self.assertEqual(actual, expected)

    def test_aggregation_functions(self):
        q = Query(mean(x) for x in "test" if x.label1 == "prod")
        a = AST(q.generators, q.groupings)
        actual = a.decompile()
        expected = FuncCall(
            Value("mean"),
            FuncArgs(
                Value([Filter(Metric("test"), CompareEqual(Value("label1"), Value("prod")))]),
                Value({}),
            ),
        )
        self.assertEqual(actual, expected)

    def test_query_with_varname_in_filter(self):
        value = "prod"
        q = Query(x for x in "test" if x.label1 == value)
        a = AST(q.generators, q.groupings)
        actual = a.decompile()
        expected = Filter(Metric("test"), CompareEqual(Value("label1"), Value("prod")))
        self.assertEqual(actual, expected)

    def test_query_with_string_variable_for_metric(self):
        value = "test"
        q = Query(x for x in value if x.label1 == "prod")
        a = AST(q.generators, q.groupings)
        actual = a.decompile()
        expected = Filter(Metric("test"), CompareEqual(Value("label1"), Value("prod")))
        self.assertEqual(actual, expected)

    def test_for_loop_filter_variables(self):
        filters = ["filter1", "filter2", "filter3"]
        for i in range(len(filters)):
            q = Query(x for x in "test" if x.label1 == filters[i])
            a = AST(q.generators, q.groupings)
            actual = a.decompile()
            expected = Filter(Metric("test"), CompareEqual(Value("label1"), Value(filters[i])))
            self.assertEqual(actual, expected)

    def test_complex_case(self):
        q = Query(
            moving_average(x, size=5) + y.mean + z
            for x, y, z in ("test", "test2", "test3")
            if z.label == "something" and z.label == "another_one"
        ).by("host")
        a = AST(q.generators, q.groupings)
        actual = a.decompile()
        expected = Group(
            BinaryAdd(
                BinaryAdd(
                    FuncCall(
                        Value("moving_average"),
                        FuncArgs(Value([Metric("test")]), Value({"size": 5})),
                    ),
                    LoadAttr(Metric("test2"), Value("mean")),
                ),
                Filter(
                    Filter(Metric("test3"), CompareEqual(Value("label"), Value("something"))),
                    CompareEqual(Value("label"), Value("another_one")),
                ),
            ),
            Value([["host"], "mean"]),
        )
        self.assertEqual(actual, expected)

    def test_node_equality(self):
        self.assertEqual(
            FuncArgs(Value([Metric("test")]), Value({"size": 5})),
            FuncArgs(Value([Metric("test")]), Value({"size": 5})),
        )
        self.assertNotEqual(
            FuncArgs(Value([Metric("test")]), Value({"size": 5})),
            FuncArgs(Value([Metric("test")]), Value({"size": 10})),
        )
        self.assertFalse(FuncArgs(Value([Metric("test")]), Value({"size": 5})) == False)

    def test_node_pprint(self):
        node = FuncArgs(Value([Metric("test")]), Value({"size": 5}))
        self.assertEqual(str(node), node.pprint(0))

    def test_value_pprint(self):
        l = [1, 2, 3, 4]
        v = Value(l).pprint(0)
        self.assertEqual("Value: [\n\t1,\n\t2,\n\t3,\n\t4,\n]", v)
        self.assertEqual(str(Value(l)), v)

    def test_value_equality(self):
        self.assertEqual(Value(5), Value(5))
        self.assertFalse(Value(5) == 5)

    def test_value_repr(self):
        v = Value(5)
        self.assertEqual(repr(v), "Value<5>")

    def test_query_for_iterator(self):
        q = Query(x for x in Query(x for x in "test"))
        a = AST(q.generators, q.groupings)
        actual = a.decompile()
        expected = Metric('test')
        self.assertEqual(actual, expected)

if __name__ == '__main__':
    unittest.main()