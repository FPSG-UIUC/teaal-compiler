from lark.lexer import Token
from lark.tree import Tree


def make_dot(tensors, rank):
    return Tree("dot", tensors + [make_num(rank)])


def make_einsum(lhs, rhs):
    return Tree("einsum", [lhs, rhs])


def make_itimes(vars_):
    return Tree("iplus", [Tree("itimes", [make_name(var) for var in vars_])])


def make_iplus(vars_):
    return Tree("iplus", [Tree("ijust", [make_name(var)]) for var in vars_])


def make_name(name):
    return Token("NAME", name)


def make_num(num):
    return Token("NUMBER", num)


def make_nway_shape(shapes):
    return [Tree("nway_shape", [make_num(shape)])
            for shape in shapes]


def make_output(name, ranks):
    return Tree("output", [make_name(name), make_tranks(ranks)])


def make_plus(vars_):
    return Tree("plus", [make_times([var]) for var in vars_])


def make_sranks(ranks):
    return Tree("sranks", [make_name(i) for i in ranks])


def make_sum(ranks, expr):
    return Tree("sum", [make_sranks(ranks), expr])


def make_tensor(name, ranks):
    return make_tensor_tranks(name, make_tranks(ranks))


def make_tensor_tranks(name, tranks):
    return Tree("tensor", [make_name(name), tranks])


def make_times(vars_):
    return Tree("times", [Tree("single", [make_var(var)]) for var in vars_])


def make_tranks(ranks):
    return Tree("tranks", [make_iplus([i]) for i in ranks])


def make_uniform_shape(shapes):
    return [Tree("uniform_shape", [make_num(shape)])
            for shape in shapes]


def make_var(var):
    return Tree("var", [make_name(var)])
