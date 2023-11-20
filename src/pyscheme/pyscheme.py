"""Pyscheme"""

import sys
import string
import tokenize
import itertools
from typing import (
    Tuple, Union, Callable, Dict, List, Iterator, Any, Optional
)


# -*-------------*-
# -*- buffer.py -*-
# -*-------------*-

class Buffer:
    """Buffer."""
    def __init__(self, source):
        self.index = 0
        self.lines = []
        self.source = source
        self.current_line = ()
        self.current()
        
    def pop(self) -> str:
        pass
    
    @property
    def has_more(self) -> bool:
        pass
    
    def current(self) -> str:
        pass
    
    def __str__(self):
        pass
    
    
class InputReader:
    """InputReader."""
    
    def __init__(self, promot:str):
        self.prompt = promot
        
    def __iter__(self):
        pass
    
class LineReader:
    """LineReader."""
    
    def __init__(self, lines, prompt, comment=";"):
        self.lines = lines
        self.prompt = prompt
        self.comment = comment
        
    def __iter__(self):
        pass


# -*- scheme_tokens.py -*-

class Tokenizer:
    """Tokenizer."""
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
    
    def __init__(self, source: str):
        self._source = source
        self._lines = source.split("\n")
        
    def is_valid_symbol(self, text:str) -> bool:
        pass
    
    def next_token(self, line:str, idx:int) -> Tuple[str, int]:
        pass
    
    def tokenize_line(self, line:str) -> List[str]:
        pass
    
    def tokenize(self) -> Iterator:
        pass
    
    def count_tokens(self):
        pass
    

# -*--------------------*-
# -*- scheme_reader.py -*-
# -*--------------------*-

class Pair:
    """Pair."""
    
    def __init__(self, first, second):
        self.first = first
        self.second = second
        
    def __repr__(self):
        pass
    
    def __str__(self):
        pass
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass
    
    def __eq__(self, other):
        pass
    
    def map(self, fun: Callable):
        pass


class Nil:
    """Nil."""
    
    _instance = None
    
    def __init__(self):
        if Nil._instance is not None:
            return
        Nil._instance = self
        
    def __repr__(self):
        pass
    
    def __str__(self):
        pass
    
    def __getitem__(self, idx):
        pass
    
    def map(self, fun: Callable):
        pass
    
nil = Nil()


# - Parser

class Parser:
    """Parser."""
    
    def __init__(self, source: Buffer):
        self._source = source
        self._tmp = None
        
    def parse(self):
        pass
    
    def _read_tail(self):
        pass
    
    @staticmethod
    def input(prompt="") -> Buffer:
        pass
    
    @staticmethod
    def lines(src:List[str], prompt="", show_prompt=False) -> Buffer:
        pass
    
    @staticmethod
    def readline(line) -> Any:
        pass
    

# -*----------------*-
# -*-- Environment -*-
# -*----------------*-

class Env:
    """Env."""
    
    def __init__(self, parent: Optional["Env"]=None):
        self.binding = {}
        self.parent = parent
        
    def __repr__(self):
        pass
    
    def lookup(self, symbol):
        pass
    
    def global_env(self):
        pass
    
    def make_call_env(self, formals, vals):
        pass
    
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