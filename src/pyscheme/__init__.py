"""Pyscheme."""
from .pyscheme import Parser
from .pyscheme import pyscm_repl
from .pyscheme import create_global_env
import sys

def main():
    print("Hello PyScheme")
    nxtline = Parser.input
    interactive = True
    load_files = ()
    argv = sys.argv[1:]
    if argv:
        try:
            name = argv[0]
            if name == "-load":
                load_files = argv[1:]
            else:
                with open(argv[0]) as fp:
                    lines = fp.readlines()
                nxtline = lambda: Parser.lines(lines)
                interactive = False
        except IOError as err:
            print(err)
            sys.exit(1)
    pyscm_repl(
        nxtline=nxtline,
        env=create_global_env(),
        startup=True,
        interactive=interactive,
        load_files=load_files
    )