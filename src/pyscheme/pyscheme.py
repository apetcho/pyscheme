"""Pyscheme"""

import sys
import math
import string
import tokenize
import itertools
import functools
from typing import (
    Tuple, Union, Callable, Dict, List, Iterator, Any, Optional
)


class PySchemeError(Exception):
    pass

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
    
    def define(self, sym, val):
        pass
    
    
class Lambda:
    """Lambda procedure."""
    
    def __init__(self, params, body, env):
        pass
    
    def __init__(self):
        pass
    
    def __repr__(self):
        pass
    
    
class Primitive:
    """Primitive."""
    
    def __init__(self, fun: Callable, use_env=False):
        pass
    
    def __str__(self):
        pass
    
    
PRIMITIVES = {}


def primitive(*names):
    """Decorator..."""
    
    def wrapper(fun):
        pass
    
def add_primitives(env: Env):
    """Ente bindings in PRIMITIVES into Env, an environment frame."""
    pass


def check_type(val, predicate, key, name):
    pass


@primitive("boolean?")
def prim_booleanp(val):
    pass


def prim_true(val):
    pass


def prim_false(val):
    pass


@primitive("not")
def prim_not(val):
    pass

@primitive("eq?", "equal?")
def prim_eqp(x, y):
    pass


@primitive("pair?")
def prim_pairp(val):
    pass


@primitive("null?")
def prim_nullp(val):
    pass


@primitive("list?")
def prim_listp(val):
    pass


@primitive("length")
def prim_length(val):
    pass


@primitive("cons")
def prim_cons(car, cdr):
    pass


@primitive("car")
def prim_car(val):
    pass


@primitive("cdr")
def prim_cdr(val):
    pass


@primitive("list")
def prim_list(*val):
    pass


@primitive("append")
def prim_append(*val):
    pass


@primitive("string?")
def prim_stringp(val):
    pass


@primitive("symbol?")
def prim_symbolp(val):
    pass


@primitive("nnumber?")
def prim_numberp(val):
    pass


@primitive("integer?")
def prim_intergerp(val):
    pass


def _check_nums(*vals):
    pass


def _arith(fun, init, vals):
    pass


@primitive("+")
def prim_add(val):
    pass


@primitive("-")
def prim_sub(val0, *vals):
    pass


@primitive("*")
def prim_mul(*val):
    pass

@primitive("/")
def prim_div(x, y):
    pass


@primitive("quotient")
def prim_quotient(x, y):
    pass


@primitive("modulo", "remainder")
def prim_modulo(x, y):
    pass


@primitive("floor")
def prim_floor(val):
    pass


@primitive("ceil")
def prim_ceil(val):
    pass


def _numcomp(op, x, y):
    pass


@primitive("=")
def prim_eq(val):
    pass


@primitive("<")
def prim_less(val):
    pass


@primitive(">")
def prim_greater(val):
    pass


@primitive("<=")
def prim_less_equal(val):
    pass


@primitive(">=")
def prim_greater_equal(val):
    pass


@primitive("even?")
def prim_evenp(val):
    pass


@primitive("odd?")
def prim_oddp(val):
    pass


@primitive("zero?")
def prim_zerop(val):
    pass


@primitive("atom?")
def prim_atomp(val):
    pass


@primitive("display")
def prim_display(val):
    pass


@primitive("print")
def prim_print(val):
    pass


@primitive("newline")
def prim_newline(val):
    pass


@primitive("error")
def prim_error(val):
    pass


@primitive("exit")
def prim_exit():
    pass


def pyscm_lambda(vals, env):
    pass


def pyscm_define(vals, env):
    pass


def pyscm_quote(vals, env):
    pass


def pyscm_let(vals, env):
    pass


def pyscm_if(vals, env):
    pass


def pyscm_and(vals, env):
    pass


def pyscm_quote(vals, env):
    pass


def pyscm_or(vals, env):
    pass


def pyscm_cond(vals, env):
    pass


def pyscm_begin(vals, env):
    pass


def check_form(expr, min, max=None):
    pass


def check_params(params):
    pass


def pyscm_optimize_eval(vals, env):
    pass


def pyscm_eval(vals, env):
    pass


def pyscm_apply(proc, args, env):
    pass


def pyscm_repl(
    nxtline, env, quiet=False, startup=False,
    interactive=False, load_files=()
) -> None:
    pass


def pyscm_load(*args):
    pass


def pyscm_open(filename):
    pass


def create_global_env(vals, env):
    pass