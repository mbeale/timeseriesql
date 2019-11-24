import dis
import collections
import types


class Node:
    left = None
    right = None

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def pprint(self, level):
        nl = "\n"
        tabs = "\t" * level
        return f"{tabs}{self.__class__.__name__}:{nl}{self.left.pprint(level+1)},{nl}{self.right.pprint(level+1)}"

    def __eq__(self, v):
        if isinstance(self, type(v)):
            return self.left == v.left and self.right == v.right
        return False

    def __str__(self):
        return self.pprint(0)


class Empty(Node):
    def __init__(self):
        self.left = None
        self.right = None

    def pprint(self, level):
        tabs = "\t" * level
        return f"{tabs}{self.__class__.__name__}:None, None"


class Filter(Node):
    pass


class Group(Node):
    pass


class CompareEqual(Node):
    pass


class CompareNotEqual(Node):
    pass


class CompareIn(Node):
    pass

class CompareNotIn(Node):
    pass

class LoadAttr(Node):
    pass


class Func(Node):
    pass


class FuncCall(Node):
    pass


class FuncArgs(Node):
    pass


class BinaryModulo(Node):
    pass


class BinaryPower(Node):
    pass


class BinaryFloorDivide(Node):
    pass


class BinaryTrueDivide(Node):
    pass


class BinaryMatrixMultiply(Node):
    pass


class BinaryMultiply(Node):
    pass


class BinaryAdd(Node):
    pass


class BinarySubtract(Node):
    pass

class BinaryXOR(Node):
    pass


class Value:
    value = None

    def __init__(self, value):
        self.value = value

    def pprint(self, level):
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
        if type(v) == type(self):
            return self.value == v.value
        return False

    def __str__(self):
        return self.pprint(0)

    def __repr__(self):
        return f"Value<{self.value}>"


class Metric(Value):
    pass


class AST:
    def __init__(self, gen, group_by=None):
        self.gen = gen
        self.group_by = group_by
        self.working_var = None
        self.variables = {}
        self.current_gen = None
        self.stack = []

    def decompile(self):
        if not isinstance(self.gen, list):
            self.gen = [self.gen]
        for gen in self.gen:
            self.current_gen = gen
            bytecode = dis.Bytecode(gen)
            for instr in bytecode:
                try:
                    op = self.__getattribute__(instr.opname.lower())
                    op(instr)
                except Exception as e:
                    raise NotImplementedError(f"opname of {instr.opname} is not supported")
            rtn = self.stack.pop()
            if self.group_by:
                rtn = Group(rtn, Value(self.group_by))
            self.stack.append(rtn)
        return rtn

    # op handlers

    def binary_add(self, instr):
        self.binary_op(BinaryAdd)

    def binary_floor_divide(self, instr):
        self.binary_op(BinaryFloorDivide)

    def binary_matrix_multiply(self, instr):
        self.binary_op(BinaryMatrixMultiply)

    def binary_multiply(self, instr):
        self.binary_op(BinaryMultiply)

    def binary_modulo(self, instr):
        self.binary_op(BinaryModulo)

    def binary_power(self, instr):
        self.binary_op(BinaryPower)

    def binary_op(self, cl):
        right = self.stack.pop()
        left = self.stack.pop()
        self.stack.append(cl(left, right))

    def binary_subscr(self, instr):
        index = self.stack.pop()
        target = self.stack.pop()

        self.stack.append(Value(target.value[index.value]))

    def binary_subtract(self, instr):
        self.binary_op(BinarySubtract)

    def binary_true_divide(self, instr):
        self.binary_op(BinaryTrueDivide)

    def binary_xor(self, instr):
        self.binary_op(BinaryXOR)

    def call_function(self, instr):
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

    def call_function_kw(self, instr):
        self.call_function(instr)

    def compare_op(self, instr):
        ops = {"in": CompareIn, "==": CompareEqual, "!=": CompareNotEqual, 'not in': CompareNotIn}
        right = self.stack.pop()
        left = self.stack.pop()
        cl = ops[instr.argval](left.right, right)
        f = Filter(left.left, cl)
        self.stack.append(f)

    def for_iter(self, instr):
        from .query import Query

        stack_val = self.stack.pop()
        if type(stack_val.value) == type(iter("")):
            self.stack.append(Metric(["".join(stack_val.value)][0]))
        elif isinstance(stack_val.value, collections.Iterable):
            try:
                for q in stack_val.value.generators:
                    if isinstance(q, Query):
                        a = AST(q, stack_val.value.groupings).decompile()
                    elif isinstance(q, types.GeneratorType):
                        a = AST(q).decompile()
                    else:
                        a = Value(q)
                    self.stack.append(a)
                if stack_val.value.groupings:
                    self.group_by = stack_val.value.groupings
            except:  # not a query object
                for q in stack_val.value:
                    self.stack.append(Metric(q))
        elif isinstance(stack_val, Query):
            self.stack.append(AST(q, stack_val.groupings).decompile())
        else:
            raise TypeError(f"Unexpected type of iterable ({type(stack_val.value)}")

    def jump_absolute(self, instr):
        pass

    def load_attr(self, instr):
        self.stack.append(LoadAttr(self.stack.pop(), Value(instr.argval)))

    def load_const(self, instr):
        if not instr.is_jump_target:
            self.stack.append(Value(instr.argval))

    def load_deref(self, instr):
        self.stack.append(Value(self.current_gen.gi_frame.f_locals.get(instr.argval, "unknown")))

    def load_fast(self, instr):
        if instr.argval.startswith("."):
            val = Value(self.current_gen.gi_frame.f_locals[instr.argval])
            self.stack.append(val)
            self.variables[instr.argval] = val
        else:
            self.working_var = instr.argval
            self.stack.append(self.variables[instr.argval])

    def load_global(self, instr):
        self.stack.append(Value(instr.argval))

    def pop_jump_if_false(self, instr):
        self.variables[self.working_var] = self.stack.pop()
        self.working_var = None

    def pop_top(self, instr):
        pass

    def return_value(self, instr):
        pass

    def store_fast(self, instr):
        self.variables[instr.argval] = self.stack.pop(0)

    def unpack_sequence(self, instr):
        pass

    def yield_value(self, instr):
        pass

