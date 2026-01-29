from osc2_parser.srunner.osc2.symbol_manager.inherits_condition_symbol import ActionInhertsSymbol
from osc2_parser.srunner.osc2.symbol_manager.qualifiedBehavior_symbol import QualifiedBehaviorSymbol
from osc2_parser.srunner.osc2.symbol_manager.scope import Scope
from osc2_parser.srunner.osc2.utils.log_manager import *


class ActionSymbol(QualifiedBehaviorSymbol, Scope):
    def __init__(self, QualifiedBehaviorSymbol):
        super().__init__(QualifiedBehaviorSymbol.name, QualifiedBehaviorSymbol.scope)

    def is_key_found(self, sym):
        if isinstance(sym, ActionInhertsSymbol):
            # Do not repeat checks on inheritance and stop recursion
            return False
        if sym.name in self.symbols and sym.name:
            return True
        if self.enclosing_scope is not None:
            return self.enclosing_scope.is_key_found(sym)
        return False
