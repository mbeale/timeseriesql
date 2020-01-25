import dis
import collections
import types


class Node:
    """
    A generic class for a tree Node.  

    params
    ------

    left: any
        the left node which will be evaluated before the right node
    right: any
        the right node evaluated after the left

    """

    left = None
    right = None

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def pprint(self, level):
        """ Pretty print for debugging purposes """
        nl = "\n"
        tabs = "\t" * level
        return f"{tabs}{self.__class__.__name__}:{nl}{self.left.pprint(level+1)},{nl}{self.right.pprint(level+1)}"

    def __eq__(self, v):
        """ Override the eq so tests can compare all children nodes as well """
        if isinstance(self, type(v)):
            return self.left == v.left and self.right == v.right
        return False

    def __str__(self):
        """ Pretty print instead of using string """
        return self.pprint(0)


class Filter(Node):
    """ Used when a filter is used in the generator """

    pass


class Group(Node):
    """ Used when a filter is used in the generator """

    pass


class CompareEqual(Node):
    """ Used when a filter is used in the generator """

    pass


class CompareNotEqual(Node):
    """ Used when a filter is used in the generator """

    pass


class CompareIn(Node):
    """ Used when a filter is used in the generator """

    pass


class CompareNotIn(Node):
    """ Used when a filter is used in the generator """

    pass


class LoadAttr(Node):
    """ Used when a filter is used in the generator """

    pass


class Func(Node):
    """ Used when a filter is used in the generator """

    pass


class FuncCall(Node):
    """ Used when a filter is used in the generator """

    pass


class FuncArgs(Node):
    """ Used when a filter is used in the generator """

    pass


class BinaryModulo(Node):
    """ Used when a filter is used in the generator """

    pass


class BinaryPower(Node):
    """ Used when a filter is used in the generator """

    pass


class BinaryFloorDivide(Node):
    """ Used when a filter is used in the generator """

    pass


class BinaryTrueDivide(Node):
    """ Used when a filter is used in the generator """

    pass


class BinaryMatrixMultiply(Node):
    """ Used when a filter is used in the generator """

    pass


class BinaryMultiply(Node):
    """ Used when a filter is used in the generator """

    pass


class BinaryAdd(Node):
    """ Used when a filter is used in the generator """

    pass


class BinarySubtract(Node):
    """ Used when a filter is used in the generator """

    pass


class BinaryXOR(Node):
    """ Used when a filter is used in the generator """

    pass


class Value:
    """ 
    A generic value node.  Using this node indicated the bottom of tree

    value: any
        The value to be stored. 
    """

    value = None

    def __init__(self, value):
        self.value = value

    def pprint(self, level):
        """ pretty print for debugging """
        tabs = "\t" * level
        val = self.value
        if isinstance(self.value, list):
            val = "[\n"
            for v in self.value:
                try:
                    val += getattr(v, "pprint")(level + 1)
                except:
                    val += tabs + "\t" + str(v)
                val += ",\n"
            val += f"{tabs}]"
        return f"{tabs}{self.__class__.__name__}: {val}"

    def __eq__(self, v):
        """ Used to compare Value objects """
        if type(v) == type(self):
            return self.value == v.value
        return False

    def __str__(self):
        return self.pprint(0)

    def __repr__(self):
        return f"Value<{self.value}>"


class Metric(Value):
    """ A class that represents the metric with the metric name as the value """

    pass


class AST:
    """
    A class to represent a generic abstract syntax to be consumed by a backend

    gen: generator(s)
        The generator(s) to be processed
    group_by: list
        A list of labels to group by
    working_var: str
        A temporary value to track which variable is currently "loaded"
    variables: map of nodes
        A map of nodes that represent the variables
    current_gen: generator
        The generator being processed
    stack: list of nodes/values
        The processing stack for the class
    """

    def __init__(self, gen, group_by=None):
        self.gen = gen
        self.group_by = group_by
        self.working_var = None
        self.variables = {}
        self.current_gen = None
        self.stack = []

    def decompile(self):
        """ The decompile function """
        if not isinstance(self.gen, list):
            self.gen = [self.gen]
        for gen in self.gen:
            self.current_gen = gen
            bytecode = dis.Bytecode(gen)
            for instr in bytecode:
                try:
                    op = self.__getattribute__("_" + instr.opname.lower())
                    op(instr)
                except:
                    raise NotImplementedError(f"opname of {instr.opname} is not supported")
            rtn = self.stack.pop()
            if self.group_by:
                rtn = Group(rtn, Value(self.group_by))
            self.stack.append(rtn)
        return rtn

    # op handlers

    def _binary_add(self, instr):
        self._binary_op(BinaryAdd)

    def _binary_floor_divide(self, instr):
        self._binary_op(BinaryFloorDivide)

    def _binary_matrix_multiply(self, instr):
        self._binary_op(BinaryMatrixMultiply)

    def _binary_multiply(self, instr):
        self._binary_op(BinaryMultiply)

    def _binary_modulo(self, instr):
        self._binary_op(BinaryModulo)

    def _binary_power(self, instr):
        self._binary_op(BinaryPower)

    def _binary_op(self, cl):
        right = self.stack.pop()
        left = self.stack.pop()
        self.stack.append(cl(left, right))

    def _binary_subscr(self, instr):
        index = self.stack.pop()
        target = self.stack.pop()

        self.stack.append(Value(target.value[index.value]))

    def _binary_subtract(self, instr):
        self._binary_op(BinarySubtract)

    def _binary_true_divide(self, instr):
        self._binary_op(BinaryTrueDivide)

    def _binary_xor(self, instr):
        self._binary_op(BinaryXOR)

    def _call_function(self, instr):
        kwargs = {}
        args = []
        arg_count = instr.arg
        if instr.opname == "CALL_FUNCTION_KW":
            kwarg_names = self.stack.pop()
            for n in reversed(kwarg_names.value):
                kwargs[n] = self.stack.pop().value
                arg_count -= 1
        for _ in range(arg_count):
            args.insert(0, self.stack.pop())
        name = self.stack.pop()
        func = FuncCall(name, FuncArgs(Value(args), Value(kwargs)))
        self.stack.append(func)

    def _call_function_kw(self, instr):
        self._call_function(instr)

    def _compare_op(self, instr):
        ops = {"in": CompareIn, "==": CompareEqual, "!=": CompareNotEqual, "not in": CompareNotIn}
        right = self.stack.pop()
        left = self.stack.pop()
        cl = ops[instr.argval](left.right, right)
        f = Filter(left.left, cl)
        self.stack.append(f)

    def _for_iter(self, instr):
        from .query import Query

        stack_val = self.stack.pop()
        if type(stack_val.value) == type(iter("")):
            self.stack.append(Metric(["".join(stack_val.value)][0]))
        else:
            try:
                for q in stack_val.value.generators:
                    a = AST(q).decompile()
                    self.stack.append(a)
                if stack_val.value.groupings:
                    self.group_by = stack_val.value.groupings
            except:  # not a query object
                # only break out if a list or tuple iter, otherwise put value on stack
                if isinstance(stack_val.value, (list, type(iter(())))):
                    for q in stack_val.value:
                        self.stack.append(Metric(q))
                else:
                    self.stack.append(Metric(stack_val.value))

    def _jump_absolute(self, instr):
        pass

    def _load_attr(self, instr):
        self.stack.append(LoadAttr(self.stack.pop(), Value(instr.argval)))

    def _load_const(self, instr):
        if not instr.is_jump_target:
            self.stack.append(Value(instr.argval))

    def _load_deref(self, instr):
        self.stack.append(Value(self.current_gen.gi_frame.f_locals.get(instr.argval, "unknown")))

    def _load_fast(self, instr):
        if instr.argval.startswith("."):
            val = Value(self.current_gen.gi_frame.f_locals[instr.argval])
            self.stack.append(val)
            self.variables[instr.argval] = val
        else:
            self.working_var = instr.argval
            self.stack.append(self.variables[instr.argval])

    def _load_global(self, instr):
        self.stack.append(
            Value(self.current_gen.gi_frame.f_globals.get(instr.argval, instr.argval))
        )

    def _pop_jump_if_false(self, instr):
        self.variables[self.working_var] = self.stack.pop()
        self.working_var = None

    def _pop_top(self, instr):
        pass

    def _return_value(self, instr):
        pass

    def _store_fast(self, instr):
        self.variables[instr.argval] = self.stack.pop(0)

    def _unpack_sequence(self, instr):
        pass

    def _yield_value(self, instr):
        pass

