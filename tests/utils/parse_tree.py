from lark.lexer import Token
from lark.tree import Tree


def make_einsum(lhs, rhs):
    return Tree("einsum", [lhs, rhs])


def make_ranks(type_, ranks):
    return Tree(type_, [Token("NAME", i) for i in ranks])


def make_dot(tensors, rank):
    return Tree("dot", tensors + [Token("NUMBER", rank)])


def make_nway_shape(shapes):
    return [Tree("nway_shape", [Token("NUMBER", shape)])
            for shape in shapes]


def make_output(name, ranks):
    return Tree("output", [Token("NAME", name), make_ranks("tranks", ranks)])


def make_plus(vars_):
    return Tree("plus", [make_times([var]) for var in vars_])


def make_sum(ranks, expr):
    return Tree("sum", [make_ranks("sranks", ranks), expr])


def make_tensor(name, ranks):
    return Tree("tensor", [Token("NAME", name), make_ranks("tranks", ranks)])


def make_times(vars_):
    return Tree("times", [Tree("single", [make_var(var)]) for var in vars_])


def make_var(var):
    return Tree("var", [Token("NAME", var)])


def make_uniform_shape(shapes):
    return [Tree("uniform_shape", [Token("NUMBER", shape)])
            for shape in shapes]
