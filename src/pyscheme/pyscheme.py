"""Pyscheme"""

import sys
import math
import string
import tokenize
import itertools
import functools
import operator
from io import FileIO
from typing import (
    Tuple, Union, Callable, Dict, List, Iterator, Any, Optional,
)


class PySchemeError(Exception):
    pass

class Okay:
    _instance = None
    def __init__(self):
        if Okay._instance is not None:
            return
        Okay._instance = self
        
    def __repr__(self):
        return "Ok"

ok = Okay()

# -*-------------*-
# -*- buffer.py -*-
# -*-------------*-

class Buffer:
    """Buffer."""
    def __init__(self, source):
        self.index = 0
        self.lines: List[str] = []
        self.source = source
        self.current_line: Optional[str] = None
        self.current()
        
    def pop(self) -> str:
        """Remove the next item from this buffer and return it."""
        current = self.current()
        self.index += 1
        return current
    
    @property
    def has_more(self) -> bool:
        return self.index < len(self.current_line)
    
    def current(self) -> Optional[str]:
        """Return the current element or None if none exists."""
        while not self.has_more:
            self.index = 0
            try:
                self.current_line = next(self.source)
                self.lines.append(self.current_line)
            except StopIteration:
                self.current_line = None
                return None
        return self.current_line[self.index]
    
    def __str__(self):
        """Return recently read contents.
        
        Current element is marked with >>.
        """
        num = len(self.lines)
        fmt = "{0:>" + str(math.floor(math.log10(num))+1) + "}: "
        txt = ""
        for i in range(max(0, num-4), num-1):
            txt += txt.format(i+1) + ' '.join(map(str, self.lines[i])) + "\n"
        txt += txt.format(num)
        txt += ' '.join(map(str, self.current_line[:self.index]))
        txt += " >> "
        txt += ' '.join(map(str, self.current_line[self.index:]))
        return txt.strip()
    
    
class InputReader:
    """InputReader."""
    
    def __init__(self, promot:str):
        self.prompt = promot
        
    def __iter__(self):
        while True:
            yield input(self.prompt)
            self.prompt = ' ' * len(self.prompt)
    
class LineReader:
    """LineReader."""
    
    def __init__(self, lines: List[str], prompt, comment=";"):
        self.lines = lines
        self.prompt = prompt
        self.comment = comment
        
    def __iter__(self):
        while self.lines:
            line = self.lines.pop(0).strip('\n')
            test = (
                self.prompt is not None and line != "" and
                not line.lstrip().startswith(self.comment)
            )
            if test:
                print(f"{self.prompt}{line}")
                self.prompt = ' ' * len(self.prompt)
            yield line
        raise EOFError


# -*- scheme_tokens.py -*-

class Tokenizer:
    """Tokenizer.
    
    A token may be:
    - A number (represented as an int or float)
    - A boolean (represented as a bool)
    - A symbol (represented as string)
    - A delimiter, including parentheses, dot, and single quotes
    """
    NUMCHARS = set(string.digits) | set("+-.")
    SYMBOLCHARS = (
        set("!$%&*/:<=>?@^_~") | set(string.ascii_letters) | NUMCHARS
    )
    STRING_DELIMITER = set('"')
    WHITESPACE_CHARS = set(" \t\n\r")
    SINGLE_TOKEN_CHARS = set("()'`")
    TOKEN_END = (
        WHITESPACE_CHARS | SINGLE_TOKEN_CHARS | STRING_DELIMITER | {",", ",@"}
    )
    DELIMITERS = SINGLE_TOKEN_CHARS | {'.', ',', ',@'}
    
    def __init__(self, source: Optional[str]=None):
        self._source = source
        self._lines = None if source is None else source.split("\n")
        
    @property
    def source(self) -> Optional[str]:
        return self._source
    
    @property
    def lines(self) -> List[str]:
        return self._lines
        
    def is_valid_symbol(self) -> bool:
        if len(self.source) == 0:
            return False
        for c in self.source:
            if c not in Tokenizer.SYMBOLCHARS:
                return False
        return True
    
    def next_token(self, line:str, idx:int) -> Tuple[Optional[str], int]:
        """A tuple (tok, idx), where tok is the next substring of line at or
        after position k that could be a token (assuming it passes a validity
        check), and idx is the position in line following that token.
        
        Returns (None, len(line)) when there are no more tokens.
        """
        while idx < len(line):
            c = line[idx]
            if c == ';':                                # comment
                return (None, len(line))
            elif c in Tokenizer.WHITESPACE_CHARS:
                idx += 1
            elif c in Tokenizer.SINGLE_TOKEN_CHARS:     # '(', ')', '\'', '`'
                return (c, idx+1)
            elif c == '#':                              # Boolean #t and #f
                return (line[idx:idx+2], min(idx+2, len(line)))
            elif c == ',':                              # Unquote: ,@
                if idx+1 < len(line) and line[idx+1] == '@':
                    return (',@', idx+2)
                return (c, idx+1)
            elif c in Tokenizer.STRING_DELIMITER:
                if idx + 1 < len(line) and line[idx+1]==c: # No triple quote
                    return (c+c, idx+2)
                linebytes = (bytes(line[idx:], encoding="utf-8"))
                token_stream = tokenize.tokenize(iter(linebytes).__next__)
                next(token_stream)   # Throw away encoding token
                token = next(token_stream)
                if token.type != tokenize.STRING:
                    raise ValueError(f"invalid string: {token.string}")
                return (token.string, token.end[1]+idx)
            else:
                i = idx
                while i < len(line) and line[i] and line[i] in Tokenizer.TOKEN_END:
                    i += 1
                return (line[idx:i], min(i, len(line)))
        return (None, len(line))
    
    def tokenize_line(self, line:str) -> List[str]:
        result = []
        tokenizer = Tokenizer()
        text, idx = tokenizer.next_token(line, 0)
        while text is not None:
            if text in Tokenizer.DELIMITERS:
                result.append(text)
            elif text=='#t' or text.lower() == "true":
                result.append(True)
            elif text=="#f" or text.lower() == "false":
                result.append(False)
            elif text=="nil":
                result.append(text)
            elif text[0] in Tokenizer.SYMBOLCHARS:
                number = False
                if text[0] in Tokenizer.NUMCHARS:
                    try:
                        result.append(int(text))        # an integer
                        number = True
                    except ValueError:
                        try:
                            result.append(float(text))  # a float
                            number = True
                        except ValueError:
                            pass
                if not number:                          # maybe symbol
                    mytknzr = Tokenizer(text)
                    if mytknzr.is_valid_symbol():       # is a symbol
                        result.append(text.lower())
                    else:                               # not a symbol
                        raise ValueError(f"invalid numeral or symbol: {text}")
            elif text[0] in Tokenizer.STRING_DELIMITER: # string
                result.append(text)
            else:                                       # Error
                print(f"warning: invalid token: {text}", file=sys.stdout)
                print(f"    {line}", file=sys.stderr)
                print(f"{' '*(idx+3)}^", file=sys.stderr)
            mytknzr = Tokenizer()
            text, idx = mytknzr.next_token(line, idx)
        return result
    
    def tokenize(self) -> Iterator:
        return map(self.tokenize_line, self.source)
    
    def count_tokens(self):
        filtered = filter(
            lambda x: x not in Tokenizer.DELIMITERS,
            itertools.chain(*self.tokenize())
        )
        return len(list(filtered))
    

# -*--------------------*-
# -*- scheme_reader.py -*-
# -*--------------------*-

class Pair:
    """Pair."""
    
    def __init__(self, car, cdr):
        self.car = car
        self.cdr = cdr
        
    def __repr__(self) -> str:
        return f"Pair({repr(self.car)}, {repr(self.cdr)})"
    
    def __str__(self) -> str:
        result = "(" + str(self.car)
        cdr = self.cdr
        while isinstance(cdr, type(self)):
            result += f" {self.car}"
            cdr = cdr.cdr
        if cdr is not nil:
            result += f" . {str(cdr)}"
        return f"{result})"
    
    def __len__(self):
        n, cdr = 1, self.cdr
        while isinstance(cdr, type(self)):
            n += 1
            cdr = cdr.cdr
        if cdr is not nil:
            raise TypeError("length attempted on improper list")
        return n
    
    def __getitem__(self, idx):
        if idx < 0:
            raise IndexError("negative index into list")
        node = self
        for _ in range(idx):
            if node.cdr is nil:
                raise IndexError("list out of bounds")
            elif not isinstance(node.cdr, type(self)):
                raise TypeError("ill-formed list")
            node = node.cdr
        return node.car
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False
        return self.car==other.car and self.cdr==other.cdr
    
    def map(self, fun: Callable):
        mapped = fun(self.car)
        if self.cdr is nil or isinstance(self.cdr, type(self)):
            return Pair(mapped, self.cdr.map(fun))
        else:
            raise TypeError("ill-formed list")


class Nil:
    """Nil."""
    
    _instance = None
    
    def __init__(self):
        if Nil._instance is not None:
            return
        Nil._instance = self
        
    def __repr__(self):
        return "nil"
    
    def __str__(self):
        return "()"
    
    def __len__(self):
        return 0
    
    def __getitem__(self, idx):
        if idx < 0:
            raise IndexError("negative index into list")
        raise IndexError("list index out of bounds")
    
    def map(self, fun: Callable):
        return self
    
nil = Nil()


# - Parser

class Parser:
    """Parser."""
    
    def __init__(self, source: Buffer):
        self._source = source
        self.tmp: Optional[Buffer] = None
        
    def read_expr(self):
        self.tmp = self._source
        if self.tmp.current() is None:
            raise EOFError
        val = self.tmp.pop()
        if val == "nil":
            return nil
        elif val not in Tokenizer.DELIMITERS:
            return val
        elif val == "'":    # quoted symbol
            pass            # TODO: add my code here
        elif val == "(":
            return self._read_tail(self.tmp)
        raise SyntaxError(f"unexpected token: {val}")
    
    def _read_tail(self):
        try:
            if self.tmp.current() is None:
                raise SyntaxError("unexpected end of file")
            if self.tmp.current() == ")":
                self.tmp.pop()
                return nil
            # TODO: add my code here
            first = self.read_expr()
            rest = self._read_tail()
            return Pair(first, rest)
        except EOFError:
            raise SyntaxError("unexpected end of file")
    
    @staticmethod
    def input(prompt="") -> Buffer:
        mytoknz = Tokenizer(InputReader(prompt))
        return Buffer(mytoknz.tokenize())
    
    @staticmethod
    def lines(src:List[str], prompt="", show_prompt=False) -> Buffer:
        if show_prompt:
            text = src
        else:
            text = LineReader(src, prompt)
        mytoknzr = Tokenizer(text)
        return Buffer(mytoknzr.tokenize())
    
    @staticmethod
    def readline(line) -> Any:
        return Parser(Buffer(Tokenizer([line]).tokenize())).read_expr()
    

# -*----------------*-
# -*-- Environment -*-
# -*----------------*-

class Env:
    """Env."""
    
    def __init__(self, parent: Optional["Env"]=None):
        self.bindings = {}
        self.parent = parent
        
    def __repr__(self):
        if self.parent is None:
            return "<GLOBAL ENVIRONMENT>"
        else:
            data = [f"{key}: {val}" for key, val in self.bindings.items()]
            result = f"<{{{', '.join(data)}}} -> {repr(self.parent)}>"
            return result
    
    def lookup(self, symbol):
        result = None
        for key, val in self.bindings.items():
            if key == symbol:
                result = val
                break
        if result is not None:
            return result
        else:
            self.parent.lookup(symbol)
        raise PySchemeError("unknown identifier")
    
    def global_env(self):
        env = self
        while env.parent is not None:
            env = env.parent
        return env
    
    def make_call_env(self, params, values):
        env = Env(self)
        # TODO : add my code here
        return env
    
    def define(self, symbol, value):
        """Define scheme symbol (symbol) to have value (value) in the current Env."""
        self.bindings[symbol] = value
    
    
class Lambda:
    """Lambda procedure."""
    __slots__= ("_params", "_body", "_env")
    
    def __init__(self, params, body, env):
        self._params = params
        self._body = body
        self._env = env
        
    @property
    def params(self):
        return self._params
    
    @property
    def body(self):
        return self._body
    
    @property
    def env(self):
        return self._env
        
    
    def __str__(self):
        return f"(lambda {str(self.params)} {str(self.body)})"
    
    def __repr__(self):
        return f"Lambda({repr(self.params)}, {repr(self.body)}, {repr(self.env)})"
    
    
class Primitive:
    """Primitive."""
    __slots__ = ("_callback", "_use_env")
    
    def __init__(self, fun: Callable, use_env=False):
        self._callback = fun
        self._use_env = use_env
                
    @property
    def callback(self):
        return self._callback
    
    @property
    def use_env(self):
        return self._use_env
    
    def __str__(self):
        return f"#<primitive>"
    
    
PRIMITIVES = {}


def primitive(*names):
    """Decorate a primitive function"""
    # functools.wraps(fun)
    def wrapper(fun):
        prim = Primitive(fun)
        for name in names:
            PRIMITIVES[name] = prim
        return fun
    return wrapper
    
def add_primitives(env: Env):
    """Ente bindings in PRIMITIVES into Env, an environment frame."""
    for name, fun in PRIMITIVES.items():
        env.define(name, fun)


def check_type(val, predicate, key, name):
    if not predicate(val):
        typename = type(val).__name__
        msg = f"argument {key} of {name} has wrong type ({typename})"
        raise PySchemeError(msg)
    return val


@primitive("boolean?")
def prim_booleanp(arg):
    return arg is True or arg is False


def prim_true(arg):
    return arg is not False


def prim_false(arg):
    return arg is False


@primitive("not")
def prim_not(arg):
    return not prim_true(arg)

@primitive("eq?", "equal?")
def prim_eqp(x, y):
    return x == y


@primitive("pair?")
def prim_pairp(arg):
    return isinstance(arg, Pair)


@primitive("null?")
def prim_nullp(arg):
    return arg is nil


@primitive("list?")
def prim_listp(arg: Union[Pair, Any]) -> bool:
    while arg is not nil:
        if not isinstance(arg, Pair):
            return False
        arg = arg.cdr
    return True


@primitive("length")
def prim_length(arg:Union[Pair, Any]) -> int:
    if arg is nil:
        return 0
    check_type(arg, prim_listp, 0, "length")
    return len(arg)


@primitive("cons")
def prim_cons(car, cdr) -> Pair:
    return Pair(car, cdr)


@primitive("car")
def prim_car(arg: Union[Pair, Any]):
    check_type(arg, prim_pairp, 0, "car")
    return arg.car


@primitive("cdr")
def prim_cdr(arg: Union[Pair, Any]):
    check_type(arg, prim_pairp, 0, "cdr")
    return arg.cdr


@primitive("list")
def prim_list(*args):
    result = nil
    for i in range(len(args)-1, -1, -1):
        result = Pair(args[i], result)
    return result


@primitive("append")
def prim_append(*args):
    if len(args) == 0:
        return nil
    result = args[-1]
    for i in range(len(args)-2, -1, -1):
        arg = args[i]
        if arg is not nil:
            check_type(arg, prim_pairp, i, "append")
            rv = node = Pair(arg.car, result)
            arg = arg.cdr
            while prim_pairp(arg):
                node.cdr = Pair(arg.cdr, result)
                node = node.cdr
                arg = node.cdr
            result = rv
    return result


@primitive("string?")
def prim_stringp(arg):
    return isinstance(arg, str) and arg.startswith('"')


@primitive("symbol?")
def prim_symbolp(arg):
    return isinstance(arg, str) and not prim_stringp(arg)


@primitive("nnumber?")
def prim_numberp(arg):
    return isinstance(arg, int) or isinstance(arg, float)


@primitive("integer?")
def prim_intergerp(arg):
    return isinstance(arg, int) or (prim_numberp(arg) and round(arg)==arg)


def _check_numbers(*args):
    for i, arg in enumerate(args):
        if not prim_numberp(arg):
            msg = f"operand {i} ({arg}) is not a number"
            raise PySchemeError(msg)


def _impl_arith(fun, init, args):
    _check_numbers(*args)
    acc = init
    for num in args:
        acc = fun(acc, num)
    if round(acc) == acc:
        acc = round(acc)
    return acc


@primitive("+")
def prim_add(*args):
    return _impl_arith(operator.add, 0, args)


@primitive("-")
def prim_sub(arg0, *args):
    if len(args) == 0:  # negation
        return -arg0
    return _impl_arith(operator.sub, arg0, args)


@primitive("*")
def prim_mul(*args):
    return _impl_arith(operator.mul, 1, args)


@primitive("/")
def prim_div(x, y):
    try:
        return _impl_arith(operator.truediv, x, [y])
    except ZeroDivisionError as err:
        raise PySchemeError(err)


@primitive("quotient")
def prim_quotient(x, y):
    try:
        return _impl_arith(operator.floordiv, x, [y])
    except ZeroDivisionError as err:
        raise PySchemeError(err)


@primitive("modulo", "remainder")
def prim_modulo(x, y):
    try:
        return _impl_arith(operator.mod, x, [y])
    except ZeroDivisionError as err:
        raise PySchemeError(err)


@primitive("floor")
def prim_floor(arg):
    _check_numbers(arg)
    return math.floor(arg)


@primitive("ceil")
def prim_ceil(arg):
    _check_numbers(arg)
    return math.ceil(arg)

# TODO: implement other basic math function including:
# cos, sin, tan, acos, asin, acos, hypot, ...


def _impl_num_cmp(op, x, y):
    _check_numbers(x, y)
    return op(x, y)


@primitive("=")
def prim_eq(x, y):
    return _impl_num_cmp(operator.eq, x, y)


@primitive("<")
def prim_less(x, y):
    return _impl_num_cmp(operator.lt, x, y)


@primitive(">")
def prim_greater(x, y):
    return _impl_num_cmp(operator.gt, x, y)


@primitive("<=")
def prim_less_equal(x, y):
    return _impl_num_cmp(operator.le, x, y)


@primitive(">=")
def prim_greater_equal(x, y):
    return _impl_num_cmp(operator.ge, x, y)


@primitive("even?")
def prim_evenp(arg):
    # TODO: arg should be an integer. check it
    _check_numbers(arg)
    return arg % 2 == 0


@primitive("odd?")
def prim_oddp(arg):
    _check_numbers(arg)
    return arg % 2 == 1


@primitive("zero?")
def prim_zerop(arg):
    _check_numbers(arg)
    return arg == 0


@primitive("atom?")
def prim_atomp(arg) -> bool:
    if prim_booleanp(arg):
        return True
    if prim_numberp(arg):
        return True
    if prim_symbolp(arg):
        return True
    if prim_nullp(arg):
        return True
    return False


@primitive("display")
def prim_display(arg) -> Okay:
    if prim_stringp(arg):
        arg = eval(arg)
    print(f"{str(arg)}", end="")
    return ok


@primitive("print")
def prim_print(arg) -> Okay:
    print(str(arg))
    return ok


@primitive("newline")
def prim_newline():
    print()
    sys.stdout.flush()
    return ok


@primitive("error")
def prim_error(message: Optional[str]=None):
    message = "" if message is None else str(message)
    raise PySchemeError(message)


@primitive("exit")
def prim_exit():
    raise EOFError


def check_form(expr, minval, maxval=None):
    if not prim_listp(expr):
        raise PySchemeError(f"Ill-formed express: {str(expr)}")
    length = len(expr)
    if length < minval:
        raise PySchemeError("Too few operands in form")
    elif maxval is not None and length > maxval:
        raise PySchemeError("Too many operand in form")


def check_params(params):
    # TODO: implement later
    pass


def pyscm_lambda(args, env):
    """Evaluate a lambda form with params `args` in environment."""
    check_form(args, 2)
    params = args[0]
    check_params(params)
    # TODO. (not completed) continue the implementation 


def pyscm_define(params, env):
    """TODO"""
    check_form(params, 2)
    symbol = params[0]
    if prim_symbolp(symbol):
        check_form(params, 2, 2)
        # TODO: (not completed) finish the implementation of this part
    elif isinstance(symbol, Pair):
        # TODO: (not completed) finish the implementation of this part
        pass
    else:
        raise PySchemeError("Invalid argument to `define'")


def pyscm_quote(params, env):
    check_form(params, 1, 1)
    # TODO: (not completed) finish the implementation of this part


def pyscm_let(params: Pair, env: Env):
    check_form(params, 2)
    bindings = params[0]
    exprs = params.cdr
    if not prim_listp(bindings):
        raise PySchemeError("Invalid bindings list in let form")
    # Add an environment containing bindings
    names, values = nil, nil
    # TODO: (not completed) finish the implementation of this part
    scope = env.make_call_env(names, values)
    # Evaluate all buf the last expression after bindings, and return the last
    last = len(exprs)-1
    for i in range(0, last):
        pyscm_eval(exprs[i], scope)
    return (exprs[last], scope)


def pyscm_if(params, env):
    check_form(params, 2, 3)
    # TODO: (not completed) finish the implementation of this part


def pyscm_and(params, env):
    # TODO: (not completed) finish the implementation of this part
    pass


def pyscm_quote(arg, env):
    return Pair("quote", Pair(arg, nil))


def pyscm_or(arg, env):
    # TODO: (not completed) finish the implementation of this part
    pass


def pyscm_cond(args, env):
    nclauses = len(args)
    for i, clause in enumerate(args):
        check_form(clause, 1)
        if clause.car == "else":
            if i < nclauses - 1:
                raise PySchemeError("else must be last")
            test = True
            if clause.cdr is nil:
                raise PySchemeError("Invalid else clause")
        else:
            test = pyscm_eval(clause.car, env)
        if prim_true(test):
            # TODO: (not completed) finish the implementation of this part
            pass
    return ok


def pyscm_begin(params, env):
    check_form(params, 1)
    # TODO: (not completed) finish the implementation of this part

LOGIC_FORMS = {
    "and": pyscm_and,
    "or": pyscm_or,
    "if": pyscm_if,
    "cond": pyscm_cond,
    "begin": pyscm_begin,
}


def pyscm_optimize_eval(expr, env: Env):
    while True:
        if expr is None:
            raise PySchemeError("Cannot evaluate an undefined expression.")
        # Evaluate atoms
        if prim_symbolp(expr):
            return env.lookup(expr)
        elif prim_atomp(expr) or prim_stringp(expr) or expr is ok:
            return expr
        # All non-atomic expressions are lists
        if not prim_listp(expr):
            raise PySchemeError(f"Malformed list: {str(expr)}")
        car, cdr = expr.car, expr.cdr
        # Evaluate combinations
        if prim_symbolp(car) and car in LOGIC_FORMS:
            # TODO: (not completed) finish the implementation of this part
            pass
        elif car == "lambda":
            return pyscm_lambda(cdr, env)
        elif car == "define":
            return pyscm_define(cdr, env)
        elif car == "quote":
            return pyscm_quote(cdr)
        elif car == "let":
            # TODO: (not completed) finish the implementation of this part
            pass
        else:
            # TODO: (not completed) finish the implementation of this part
            pass


def pyscm_eval(expr: Optional[Any], env: Env):
    if expr is None:
        raise PySchemeError("cannot evaluate an undefined expression.")
    # Evaluate atoms
    if prim_symbolp(expr):
        return env.lookup(expr)
    elif prim_atomp(expr) or prim_stringp(expr) or expr is ok:
        return expr
    # All non-atomic expressions are lists
    if not prim_listp(expr):
        raise PySchemeError(f"Malformed list: {str(expr)}")
    car, cdr = expr.car, expr.cdr
    # Evaluate combinations
    if prim_symbolp(car) and cdr in LOGIC_FORMS:
        return pyscm_eval(LOGIC_FORMS[car](cdr, env), env)
    elif car == "lambda":
        return pyscm_lambda(cdr, env)
    elif car == "define":
        return pyscm_define(cdr, env)
    elif car == "quote":
        return pyscm_quote(cdr)
    elif car == "let":
        expr, env = pyscm_let(cdr, env)
        return pyscm_eval(expr, env)
    else:
        fun = pyscm_eval(car, env)
        args = cdr.map(lambda xarg: pyscm_eval(xarg, env))
        return pyscm_apply(fun, args, env)


def apply_primitive(fun, args, env):
    # TODO: (not completed) finish the implementation of this part
    pass


def pyscm_apply(fun, args, env):
    if isinstance(fun, Primitive):
        return apply_primitive(fun, args, env)
    elif isinstance(fun, Lambda):
        # TODO: (not completed) finish the implementation of this part
        pass
    else:
        raise PySchemeError(f"Cannot call {str(fun)}")


def pyscm_repl(
    nxtline: Callable, env, quiet=False, startup=False,
    interactive=False, load_files=()
) -> None:
    """Read and evaluate input until and end-of-line or keyboard interrupt."""
    if startup:
        for filename in load_files:
            pyscm_load(filename, True, env)
    while True:
        try:
            buffer: Buffer = nxtline()
            while buffer.has_more:
                expr = Parser(buffer).read_expr()
                result = pyscm_eval(expr, env)
                if not quiet and result is not None:
                    print(result)
        except (PySchemeError, SyntaxError, ValueError, RuntimeError) as err:
            txt = "maximum recursion depth exceeded"
            if isinstance(err, RuntimeError) and txt not in err.args[0]:
                raise
            print(f"Error: {err}")
        except KeyboardInterrupt:   # <Control>-C
            if not startup:
                raise
            print("\nKeyboardInterrupt")
            if not interactive:
                return
        except EOFError:            # <Control>-D
            return


def pyscm_load(*args):
    if not (2 <= len(args) and len(args) <=3):
        vals = args[:-1]
        raise PySchemeError(f"Wron number of arguments to load: {vals}")
    symbol = args[0]
    quiet = args[1] if len(args) > 1 else True
    env: Env = args[-1]
    if prim_stringp(symbol):
        symbol = eval(symbol)
    check_type(symbol, prim_symbolp, 0, "load")
    with pyscm_open(symbol) as fp:
        lines = fp.readlines()
    args = (lines, None) if quiet else (lines, )
    nxtline = lambda: Parser.lines(args)
    pyscm_repl(nxtline=nxtline, env=env.global_env(), quiet=quiet)
    return ok


def pyscm_open(filename: str) -> FileIO:
    try:
        return open(filename)
    except IOError as err:
        if filename.endswith(".scm"):
            raise PySchemeError(str(err))
        
    try:
        return open(f"{filename}.scm")
    except IOError as err:
        raise PySchemeError(str(err))


def create_global_env() -> Env:
    env = Env(None)
    env.define("eval", Primitive(pyscm_eval, True))
    env.define("apply", Primitive(pyscm_apply, True))
    env.define("load", Primitive(pyscm_load, True))
    env.define("import", Primitive(pyscm_load, True))
    add_primitives(env)
    return env