from __future__ import annotations
import dis
from typing import Any, Callable, Dict, Generator, Optional, List, Union


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

    left: Union[Node, Value]
    right: Union[Node, Value]

    def __init__(self, left: Union[Node, Value], right: Union[Node, Value]):
        self.left = left
        self.right = right

    def pprint(self, level: int) -> str:
        """ Pretty print for debugging purposes """
        nl: str = "\n"
        tabs: str = "\t" * level
        return f"{tabs}{self.__class__.__name__}:{nl}{self.left.pprint(level+1)},{nl}{self.right.pprint(level+1)}"

    def __eq__(self, other: object) -> bool:
        """ Override the eq so tests can compare all children nodes as well """
        if not isinstance(other, Node):
            return NotImplemented
        return self.left == other.left and self.right == other.right

    def __str__(self) -> str:
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

    value: Any = None

    def __init__(self, value):
        self.value = value

    def pprint(self, level: int) -> str:
        """ pretty print for debugging """
        tabs: str = "\t" * level
        val: Any = self.value
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

    def __eq__(self, other: object):
        """ Used to compare Value objects """
        if not isinstance(other, Value):
            return NotImplemented
        return self.value == other.value

    def __str__(self) -> str:
        return self.pprint(0)

    def __repr__(self) -> str:
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

    gen: Union[List[Generator], Generator]
    current_gen: Optional[Generator]
    stack: List[Union[Node, Value]]
    group_by: Optional[List[str]]
    variables: Dict[str, Any]
    working_var: Optional[str]

    def __init__(self, gen, group_by=None):
        self.gen = gen
        self.group_by = group_by
        self.working_var = None
        self.variables = {}
        self.current_gen = None
        self.stack = []

    def decompile(self) -> Union[Node, Value]:
        """ The decompile function """
        if not isinstance(self.gen, list):
            self.gen = [self.gen]
        rtn: Union[Node, Value] = Value(None)
        for gen in self.gen:
            self.current_gen = gen
            bytecode: dis.Bytecode = dis.Bytecode(gen)  # type: ignore
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

    def _binary_add(self, instr: dis.Instruction):
        self._binary_op(BinaryAdd)

    def _binary_floor_divide(self, instr: dis.Instruction):
        self._binary_op(BinaryFloorDivide)

    def _binary_matrix_multiply(self, instr: dis.Instruction):
        self._binary_op(BinaryMatrixMultiply)

    def _binary_multiply(self, instr: dis.Instruction):
        self._binary_op(BinaryMultiply)

    def _binary_modulo(self, instr: dis.Instruction):
        self._binary_op(BinaryModulo)

    def _binary_power(self, instr: dis.Instruction):
        self._binary_op(BinaryPower)

    def _binary_op(self, cl):
        right: Union[Node, Value] = self.stack.pop()
        left: Union[Node, Value] = self.stack.pop()
        self.stack.append(cl(left, right))

    def _binary_subscr(self, instr: dis.Instruction):
        index: Union[Node, Value] = self.stack.pop()
        target: Union[Node, Value] = self.stack.pop()

        if not isinstance(index, Value):
            return NotImplemented
        if not isinstance(target, Value):
            return NotImplemented
        self.stack.append(Value(target.value[index.value]))

    def _binary_subtract(self, instr: dis.Instruction):
        self._binary_op(BinarySubtract)

    def _binary_true_divide(self, instr: dis.Instruction):
        self._binary_op(BinaryTrueDivide)

    def _binary_xor(self, instr: dis.Instruction):
        self._binary_op(BinaryXOR)

    def _call_function(self, instr: dis.Instruction):
        kwargs: Dict[Any, Any] = {}
        args: List[Union[Node, Value]] = []
        arg_count: int = 0 if not instr.arg else instr.arg
        if instr.opname == "CALL_FUNCTION_KW":
            kwarg_names: Union[Node, Value] = self.stack.pop()
            if not isinstance(kwarg_names, Value):
                return NotImplemented
            for n in reversed(kwarg_names.value):
                value: Union[Node, Value] = self.stack.pop()
                if not isinstance(value, Value):
                    return NotImplemented
                kwargs[n] = value.value
                arg_count -= 1
        for _ in range(arg_count):
            args.insert(0, self.stack.pop())
        name: Union[Node, Value] = self.stack.pop()
        func: FuncCall = FuncCall(name, FuncArgs(Value(args), Value(kwargs)))
        self.stack.append(func)

    def _call_function_kw(self, instr: dis.Instruction):
        self._call_function(instr)

    def _compare_op(self, instr: dis.Instruction):
        ops: Dict[str, Callable] = {
            "in": CompareIn,
            "==": CompareEqual,
            "!=": CompareNotEqual,
            "not in": CompareNotIn,
        }
        right: Union[Node, Value] = self.stack.pop()
        left: Union[Node, Value] = self.stack.pop()
        if not isinstance(left, Node):
            return NotImplemented
        cl: Node = ops[instr.argval](left.right, right)
        f: Node = Filter(left.left, cl)
        self.stack.append(f)

    def _for_iter(self, instr: dis.Instruction):
        from .query import Query

        stack_val: Union[Node, Value] = self.stack.pop()
        if not isinstance(stack_val, Value):
            return NotImplemented
        if type(stack_val.value) == type(iter("")):
            self.stack.append(Metric(["".join(stack_val.value)][0]))
        else:
            try:
                for q in stack_val.value.generators:
                    a: Union[Node, Value] = AST(q).decompile()
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

    def _jump_absolute(self, instr: dis.Instruction):
        pass

    def _load_attr(self, instr: dis.Instruction):
        self.stack.append(LoadAttr(self.stack.pop(), Value(instr.argval)))

    def _load_const(self, instr: dis.Instruction):
        if not instr.is_jump_target:
            self.stack.append(Value(instr.argval))

    def _load_deref(self, instr: dis.Instruction):
        if not isinstance(self.current_gen, Generator):
            raise TypeError("Expecting a cuurrent generator but instead it's set to None")
        self.stack.append(Value(self.current_gen.gi_frame.f_locals.get(instr.argval, "unknown")))

    def _load_fast(self, instr: dis.Instruction):
        if instr.argval.startswith("."):
            if not isinstance(self.current_gen, Generator):
                raise TypeError("Expecting a cuurrent generator but instead it's set to None")
            val: Value = Value(self.current_gen.gi_frame.f_locals[instr.argval])
            self.stack.append(val)
            self.variables[instr.argval] = val
        else:
            self.working_var = instr.argval
            self.stack.append(self.variables[instr.argval])

    def _load_global(self, instr: dis.Instruction):
        if not isinstance(self.current_gen, Generator):
            raise TypeError("Expecting a cuurrent generator but instead it's set to None")
        self.stack.append(
            Value(self.current_gen.gi_frame.f_globals.get(instr.argval, instr.argval))
        )

    def _pop_jump_if_false(self, instr: dis.Instruction):
        if not self.working_var:
            return NotImplemented
        self.variables[self.working_var] = self.stack.pop()
        self.working_var = None

    def _pop_top(self, instr: dis.Instruction):
        pass

    def _return_value(self, instr: dis.Instruction):
        pass

    def _store_fast(self, instr: dis.Instruction):
        self.variables[instr.argval] = self.stack.pop(0)

    def _unpack_sequence(self, instr: dis.Instruction):
        pass

    def _yield_value(self, instr: dis.Instruction):
        pass
