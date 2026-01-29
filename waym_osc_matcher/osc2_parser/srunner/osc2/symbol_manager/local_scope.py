import copy
from collections import defaultdict

from osc2_parser.srunner.osc2.symbol_manager.base_scope import BaseScope
from osc2_parser.srunner.osc2.utils.log_manager import *
from osc2_parser.srunner.osc2.utils.tools import is_multi_name, multi_field_name_split


class LocalScope(BaseScope):
    def __init__(self, scope):
        super().__init__(scope)
        self.symbols = {}
        self.redefinitions = defaultdict(list)

    # For local scopes, only internal naming conflicts are found
    def is_key_found(self, sym):
        return bool(sym.name and sym.name in self.symbols)

    def _attach_backref(self, sym):
        if hasattr(sym, "set_enclosing_scope"):
            sym.set_enclosing_scope(self)
        else:
            try:
                sym.enclosing_scope = self
            except Exception:
                pass

    def define(self, sym, ctx):
        # treat LocalScope instances as anonymous inner-scope entries
        if isinstance(sym, LocalScope):
            key = f"{sym.__class__.__name__}@{getattr(ctx, 'line', '?')}:{getattr(ctx, 'column', '?')}"
            self.symbols[key] = sym
            # it already has its enclosing scope; keep as-is
            return sym

        # handle multi-name expansion (e.g., "a,b,c")
        if is_multi_name(sym.name):
            canonical = None
            parts = multi_field_name_split(sym.name)
            for sub in parts:
                sub_sym = copy.deepcopy(sym)
                sub_sym.name = sub
                if sub in self.symbols:
                    # keep first; record later ones
                    self.redefinitions[sub].append((sub_sym, ctx))
                else:
                    self.symbols[sub] = sub_sym
                    self._attach_backref(sub_sym)
                    if canonical is None:
                        canonical = sub_sym
            # return the canonical (first defined) symbol for chaining
            return canonical or self.symbols[parts[0]]

        # single-name path (strip composite suffix if you use '#')
        base = sym.name.split("#", 1)[0]
        if base in self.symbols:
            # keep the first; record this one
            self.redefinitions[base].append((sym, ctx))
            return self.symbols[base]
        else:
            self.symbols[base] = sym
            self._attach_backref(sym)
            return sym

    def get_scope_name(self):
        return "local"