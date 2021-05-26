from lark.lexer import Token
from lark.tree import Tree


def make_inds(type_, inds):
    return Tree(type_, [Token("NAME", i) for i in inds])


def make_op(var1, op, var2):
    return Tree(op, [Token("NAME", var1), Token("NAME", var2)])


def make_output(name, inds):
    return Tree("output", [Token("NAME", name), make_inds("tinds", inds)])


def make_einsum(lhs, rhs):
    return Tree("einsum", [lhs, rhs])


def make_tensor(name, inds):
    return Tree("tensor", [Token("NAME", name), make_inds("tinds", inds)])
