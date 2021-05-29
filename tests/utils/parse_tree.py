from lark.lexer import Token
from lark.tree import Tree


def make_einsum(lhs, rhs):
    return Tree("einsum", [lhs, rhs])


def make_inds(type_, inds):
    return Tree(type_, [Token("NAME", i) for i in inds])


def make_output(name, inds):
    return Tree("output", [Token("NAME", name), make_inds("tinds", inds)])


def make_plus(var1, var2):
    return Tree("plus", [make_times([var1]), make_times([var2])])


def make_tensor(name, inds):
    return Tree("tensor", [Token("NAME", name), make_inds("tinds", inds)])


def make_times(vars_):
    return Tree("times", [make_var(var) for var in vars_])


def make_var(var):
    return Tree("var", [Token("NAME", var)])
