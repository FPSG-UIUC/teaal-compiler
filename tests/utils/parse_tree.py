from lark.lexer import Token
from lark.tree import Tree


def make_take(tensors, rank):
    return Tree("take", tensors + [make_num(rank)])


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
    return Tree("output", [make_name(name), make_ranks(ranks)])


def make_plus(vars_):
    return Tree("plus", [make_times([var]) for var in vars_])


def make_prank(rank):
    return Tree("rank", [Token("NAME", rank)])


def make_pranks(ranks):
    return Tree("ranks", [Token("NAME", rank) for rank in ranks])


def make_tensor(name, ranks):
    return make_tensor_ranks(name, make_ranks(ranks))


def make_tensor_ranks(name, ranks):
    return Tree("tensor", [make_name(name), ranks])


def make_times(vars_):
    return Tree("times", [make_var(var) for var in vars_])


def make_ranks(ranks):
    return Tree("ranks", [make_iplus([i]) for i in ranks])


def make_uniform_shape(shapes):
    return [Tree("uniform_shape", [Tree("int_sz", [make_num(shape)])])
            for shape in shapes]


def make_var(var):
    return Tree("var", [make_name(var)])
