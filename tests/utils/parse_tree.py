from lark.lexer import Token
from lark.tree import Tree


def make_einsum(lhs, rhs):
    return Tree("einsum", [lhs, rhs])


def make_inds(type_, inds):
    return Tree(type_, [Token("NAME", i) for i in inds])


def make_dot(tensors, ind):
    return Tree("dot", tensors + [Token("NUMBER", ind)])


def make_nway_shape(shapes):
    return [Tree("nway_shape", [Token("NUMBER", shape)])
            for shape in shapes]


def make_output(name, inds):
    return Tree("output", [Token("NAME", name), make_inds("tinds", inds)])


def make_plus(vars_):
    return Tree("plus", [make_times([var]) for var in vars_])


def make_sum(inds, expr):
    return Tree("sum", [make_inds("sinds", inds), expr])


def make_tensor(name, inds):
    return Tree("tensor", [Token("NAME", name), make_inds("tinds", inds)])


def make_times(vars_):
    return Tree("times", [Tree("single", [make_var(var)]) for var in vars_])


def make_var(var):
    return Tree("var", [Token("NAME", var)])


def make_uniform_shape(shapes):
    return [Tree("uniform_shape", [Token("NUMBER", shape)])
            for shape in shapes]
