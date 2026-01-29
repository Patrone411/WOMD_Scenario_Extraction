import copy

from osc2_parser.srunner.osc2.symbol_manager.local_scope import LocalScope
from osc2_parser.srunner.osc2.symbol_manager.symbol import Symbol
from osc2_parser.srunner.osc2.utils.log_manager import *
from osc2_parser.srunner.osc2.utils.tools import *


class BaseSymbol(Symbol):
    def __init__(self, name, scope):
        super().__init__(name, scope)
        self.enclosing_scope = scope
        self.symbols = {}

    def resolve(self, name):
        s = self.symbols.get(name)
        if s is not None:
            return s
        if self.enclosing_scope is not None:
            return self.enclosing_scope.resolve(name)
        return None

    def is_key_found(self, sym):
        if sym.name in self.symbols and sym.name:
            return True
        if self.enclosing_scope is not None:
            return self.enclosing_scope.is_key_found(sym)
        return False

    def define(self, sym, ctx):
        # Make sure we have a place to record duplicates for the later pass
        if not hasattr(self, "redefinitions"):
            self.redefinitions = {}

        # 1) Local scopes get a unique synthetic key; return the scope itself
        try:
            from .local_scope import LocalScope  # or wherever LocalScope is defined
        except Exception:
            LocalScope = type("LocalScopeStub", (), {})  # fallback if import cycles

        if isinstance(sym, LocalScope) or issubclass(type(sym), LocalScope):
            key = f"{sym.__class__.__name__}@{getattr(ctx, 'line', None)}:{getattr(ctx, 'column', None)}"
            self.symbols[key] = sym
            return sym

        # 2) Multi-name (e.g., "a,b") – define each sub-symbol; return the first canonical
        if is_multi_name(sym.name):
            names = multi_field_name_split(sym.name)
            first_inserted = None
            for sub_name in names:
                sub_sym = copy.deepcopy(sym)
                sub_sym.name = sub_name
                if self.is_key_found(sub_sym):
                    # record, don't overwrite; defer erroring to the post-pass
                    self.redefinitions.setdefault(sub_name, []).append((sub_sym, ctx))
                else:
                    self.symbols[sub_name] = sub_sym
                    if first_inserted is None:
                        first_inserted = sub_sym
            # return the inserted canonical if any; otherwise return the existing canonical
            return first_inserted or self.symbols.get(names[0])

        # 3) Single name
        if self.is_key_found(sym):
            # record the redefinition and return the existing canonical symbol
            self.redefinitions.setdefault(sym.name, []).append((sym, ctx))
            return self.symbols[sym.name]
        else:
            self.symbols[sym.name] = sym
            return sym

    def get_enclosing_scope(self):
        return self.enclosing_scope

    def get_number_of_symbols(self):
        return len(self.symbols)

    def get_child_symbol(self, i):
        return list(self.symbols.values())[i]

    def __str__(self):
        buf = self.__class__.__name__
        buf += " : "
        buf += self.name
        return buf
