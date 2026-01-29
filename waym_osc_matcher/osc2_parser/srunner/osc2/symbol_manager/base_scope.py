from osc2_parser.srunner.osc2.symbol_manager.scope import Scope
from osc2_parser.srunner.osc2.symbol_manager.symbol import Symbol
from osc2_parser.srunner.osc2.utils.log_manager import *
from collections import defaultdict

class BaseScope(Scope):
    def __init__(self, scope: Scope):
        self.enclosing_scope = scope
        self.symbols = {}
        self.redefinitions = defaultdict(list)

    def resolve(self, name):
        s = self.symbols.get(name)
        if s is not None:
            return s
        if self.enclosing_scope is not None:
            return self.enclosing_scope.resolve(name)
        return None

    # Make this LOCAL-ONLY (optional but recommended).
    def is_key_found(self, sym):
        return bool(sym.name and sym.name in self.symbols)

    def define(self, sym, ctx):
        """
        Keep the first symbol as canonical. Subsequent same-name symbols are
        collected for a later namespace-aware check. No errors/logging here.
        Returns the canonical symbol so callers can always keep working.
        """
        name = sym.name
        if name in self.symbols:
            self.redefinitions.setdefault(name, []).append((sym, ctx))
            return self.symbols[name]  # always return canonical

        self.symbols[name] = sym
        # set backref so later code relying on get_enclosing_scope() works
        if hasattr(sym, "set_enclosing_scope"):
            sym.set_enclosing_scope(self)
        else:
            try:
                sym.enclosing_scope = self
            except Exception:
                pass
        return sym

    def get_enclosing_scope(self):
        return self.enclosing_scope

    def get_number_of_symbols(self):
        return len(self.symbols)

    def get_child_symbol(self, i):
        return list(self.symbols.values())[i]

    def __str__(self):
        # guard if get_scope_name() isn't implemented upstream
        scope_name = getattr(self, "get_scope_name", lambda: self.__class__.__name__)()
        return scope_name + " : " + list(self.symbols.keys()).__str__()