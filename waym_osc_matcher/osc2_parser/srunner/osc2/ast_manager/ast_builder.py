import copy
from pydoc import resolve

from . import ast_node
from osc2_parser.srunner.osc2.osc2_parser.OpenSCENARIO2Listener import OpenSCENARIO2Listener
from osc2_parser.srunner.osc2.osc2_parser.OpenSCENARIO2Parser import OpenSCENARIO2Parser
from osc2_parser.srunner.osc2.symbol_manager.action_symbol import ActionSymbol
from osc2_parser.srunner.osc2.symbol_manager.actor_symbol import ActorSymbol, ActionInhertsSymbol
from osc2_parser.srunner.osc2.symbol_manager.argument_symbol import *
from osc2_parser.srunner.osc2.symbol_manager.constraint_decl_scope import *
from osc2_parser.srunner.osc2.symbol_manager.do_directive_scope import *
from osc2_parser.srunner.osc2.symbol_manager.doMember_symbol import DoMemberSymbol
from osc2_parser.srunner.osc2.symbol_manager.enum_symbol import *
from osc2_parser.srunner.osc2.symbol_manager.event_symbol import *
from osc2_parser.srunner.osc2.symbol_manager.global_scope import GlobalScope
from osc2_parser.srunner.osc2.symbol_manager.inherits_condition_symbol import *
from osc2_parser.srunner.osc2.symbol_manager.method_symbol import MethodSymbol
from osc2_parser.srunner.osc2.symbol_manager.modifier_symbol import *
from osc2_parser.srunner.osc2.symbol_manager.parameter_symbol import ParameterSymbol
from osc2_parser.srunner.osc2.symbol_manager.physical_type_symbol import PhysicalTypeSymbol
from osc2_parser.srunner.osc2.symbol_manager.qualifiedBehavior_symbol import QualifiedBehaviorSymbol
from osc2_parser.srunner.osc2.symbol_manager.scenario_symbol import ScenarioSymbol, ScenarioInhertsSymbol
from osc2_parser.srunner.osc2.symbol_manager.si_exponent_symbol import (
    SiBaseExponentListScope,
    SiExpSymbol,
)
from osc2_parser.srunner.osc2.symbol_manager.struct_symbol import StructSymbol
from osc2_parser.srunner.osc2.symbol_manager.typed_symbol import *
from osc2_parser.srunner.osc2.symbol_manager.unit_symbol import UnitSymbol
from osc2_parser.srunner.osc2.symbol_manager.variable_symbol import VariableSymbol
from osc2_parser.srunner.osc2.symbol_manager.wait_symbol import *
from osc2_parser.srunner.osc2.utils.log_manager import *
from osc2_parser.srunner.osc2.utils.tools import *


class ASTBuilder(OpenSCENARIO2Listener):
    def __init__(self):
        self.__global_scope = None  # Global scope
        self.__current_scope = None  # Current scope
        self.__node_stack = []
        self.__cur_node = None
        self.ast = None

    def _define_here(self, sym, ctx):
        # define in current scope, but do not enter it
        return self.__current_scope.define(sym, ctx.start)

    def _define_and_enter(self, sym, ctx):
        """
        Define 'sym' in current scope and enter the canonical symbol/scope.
        Some define(...) implementations may return None; fall back to 'sym'.
        """
        canon = None
        try:
            canon = self.__current_scope.define(sym, ctx.start)
        except Exception:
            canon = None
        self.__current_scope = canon or sym
        return self.__current_scope

    # ---------- small helpers (safe parent + uniform node attach) ----------
    def _push_parent(self, ctx):
        """Safe push of current node; returns previous parent."""
        parent = self.__cur_node
        self.__node_stack.append(parent)
        return parent

    def _pop_parent(self, ctx):
        """Safe pop back to previous node; warns on underflow."""
        if self.__node_stack:
            self.__cur_node = self.__node_stack.pop()
        else:
            LOG_WARNING("node_stack underflow; ignoring extra exit.", getattr(ctx, "start", None))

    def _open_node(self, ctx, node, scope=None):
        """Attach node under current cursor with location+scope set, advance cursor."""
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(scope if scope is not None else self.__current_scope)
        self.__cur_node.set_children(node)
        self.__cur_node = node

    def _define_or_reuse(self, scope, name, build_sym, ctx, expected_cls=None):
        """
        Reuse existing symbol 'name' in 'scope' if present (and optionally of expected class),
        otherwise build & define a new one via build_sym(). Return canonical if provided.
        """
        existing = getattr(scope, "symbols", {}).get(name)
        if existing is not None and (expected_cls is None or isinstance(existing, expected_cls)):
            return existing
        sym = build_sym()
        canon = None
        try:
            canon = scope.define(sym, ctx.start)
        except Exception:
            canon = None
        return canon or sym

    def _owner_for_actor_qualified(self, actor_name: str, ctx):
        """Return the owning scope for an actor-qualified declaration."""
        if not actor_name:
            if isinstance(self.__current_scope, ActorSymbol):
                return self.__current_scope
            return self.__global_scope
        owner = self.__resolve_global_symbol(actor_name)
        if not isinstance(owner, ActorSymbol):
            LOG_WARNING(f"actorName: {actor_name} not resolved; using global as owner.", ctx.start)
            return self.__global_scope
        return owner

    def get_ast(self):
        return self.ast

    def get_symbol(self):
        return self.__current_scope

    def __ensure_scope(self, ctx):
        if self.__current_scope is None:
            # Recover gracefully; keeps parsing instead of crashing.
            self.__current_scope = self.__global_scope
            LOG_WARNING("current_scope was None; defaulting to global scope.", ctx.start)

    def _leave_scope(self, ctx):
        """Safely leave current scope; tolerate None + missing enclosing scope."""
        if self.__current_scope is None:
            LOG_WARNING("current_scope None on exit; defaulting to global.", ctx.start)
            self.__current_scope = self.__global_scope
            return
        try:
            enc = self.__current_scope.get_enclosing_scope()
        except Exception:
            enc = None
        self.__current_scope = enc or self.__global_scope

    # --- helpers for qualified names / actor scope ---
    def __split_qualified(self, qname: str):
        parts = qname.split(".", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            return parts[0], parts[1]
        return None, qname

    def __resolve_global(self, qualified_name: str):
        """Resolve names like 'Actor.Action' from the global scope."""
        if not qualified_name or not self.__global_scope:
            return None

        parts = qualified_name.split(".", 1)
        if len(parts) == 2:
            actor_name, local = parts
            actor_scope = self.__resolve_global_symbol(actor_name)
            if not actor_scope:
                return None

            cand = getattr(actor_scope, "symbols", {}).get(local)
            if cand:
                return cand

            cand = getattr(actor_scope, "symbols", {}).get(qualified_name)
            if cand:
                return cand

            try:
                cand = actor_scope.resolve(local)
                if cand:
                    return cand
                return actor_scope.resolve(qualified_name)
            except Exception:
                return None

        try:
            return self.__global_scope.resolve(qualified_name)
        except Exception:
            return getattr(self.__global_scope, "symbols", {}).get(qualified_name)

    def __is_actor_name_known(self, name: str) -> bool:
        from osc2_parser.srunner.osc2.symbol_manager.actor_symbol import ActorSymbol
        from osc2_parser.srunner.osc2.symbol_manager.variable_symbol import VariableSymbol
        from osc2_parser.srunner.osc2.symbol_manager.parameter_symbol import ParameterSymbol

        g = self.__resolve_global(name)
        if isinstance(g, ActorSymbol):
            return True

        cur = None
        try:
            cur = self.__current_scope.resolve(name)
        except Exception:
            pass
        if cur is None:
            cur = getattr(self.__current_scope, "symbols", {}).get(name)

        if isinstance(cur, ActorSymbol):
            return True

        if isinstance(cur, (VariableSymbol, ParameterSymbol)):
            tname = getattr(cur, "type", None)
            if tname and isinstance(self.__resolve_global(tname), ActorSymbol):
                return True

        return False

    def __resolve_global_symbol(self, name: str):
        if not self.__global_scope:
            return None
        sym = getattr(self.__global_scope, "symbols", {}).get(name)
        if sym is not None:
            return sym
        try:
            return self.__global_scope.resolve(name)
        except Exception:
            return None

    def __ensure_actor_defined(self, actor_name: str, token):
        sym = self.__resolve_global_symbol(actor_name)
        if not (sym and isinstance(sym, ActorSymbol)):
            LOG_ERROR(f"actorName: {actor_name} is not defined!", token)

    # Enter a parse tree produced by OpenSCENARIO2Parser#osc_file.
    def enterOsc_file(self, ctx: OpenSCENARIO2Parser.Osc_fileContext):
        self.__global_scope = GlobalScope(None)
        self.__current_scope = self.__global_scope

        node = ast_node.CompilationUnit()
        node.set_loc(ctx.start.line, ctx.start.column)
        node.set_scope(self.__current_scope)
        self.__node_stack.append(node)
        self.__cur_node = node
        self.ast = node

    def exitOsc_file(self, ctx: OpenSCENARIO2Parser.Osc_fileContext):
        pass

    def enterPreludeStatement(self, ctx: OpenSCENARIO2Parser.PreludeStatementContext):
        pass

    def exitPreludeStatement(self, ctx: OpenSCENARIO2Parser.PreludeStatementContext):
        pass

    def enterImportStatement(self, ctx: OpenSCENARIO2Parser.ImportStatementContext):
        pass

    def exitImportStatement(self, ctx: OpenSCENARIO2Parser.ImportStatementContext):
        pass

    def enterImportReference(self, ctx: OpenSCENARIO2Parser.ImportReferenceContext):
        pass

    def exitImportReference(self, ctx: OpenSCENARIO2Parser.ImportReferenceContext):
        pass

    def enterStructuredIdentifier(self, ctx: OpenSCENARIO2Parser.StructuredIdentifierContext):
        pass

    def exitStructuredIdentifier(self, ctx: OpenSCENARIO2Parser.StructuredIdentifierContext):
        pass

    def enterOscDeclaration(self, ctx: OpenSCENARIO2Parser.OscDeclarationContext):
        pass

    def exitOscDeclaration(self, ctx: OpenSCENARIO2Parser.OscDeclarationContext):
        pass

    # -------- Physical types / units --------
    def enterPhysicalTypeDeclaration(self, ctx: OpenSCENARIO2Parser.PhysicalTypeDeclarationContext):
        self.__ensure_scope(ctx)
        parent = self._push_parent(ctx)
        type_name = ctx.physicalTypeName().getText()

        physical_type = self._define_or_reuse(
            self.__current_scope, type_name,
            lambda: PhysicalTypeSymbol(type_name, self.__current_scope),
            ctx, expected_cls=PhysicalTypeSymbol
        )
        self.__current_scope = physical_type

        node = ast_node.PhysicalTypeDeclaration(type_name)
        self._open_node(ctx, node, self.__current_scope)

    def exitPhysicalTypeDeclaration(self, ctx: OpenSCENARIO2Parser.PhysicalTypeDeclarationContext):
        self._pop_parent(ctx)
        self._leave_scope(ctx)

    def enterPhysicalTypeName(self, ctx: OpenSCENARIO2Parser.PhysicalTypeNameContext):
        pass

    def exitPhysicalTypeName(self, ctx: OpenSCENARIO2Parser.PhysicalTypeNameContext):
        pass

    def enterBaseUnitSpecifier(self, ctx: OpenSCENARIO2Parser.BaseUnitSpecifierContext):
        pass

    def exitBaseUnitSpecifier(self, ctx: OpenSCENARIO2Parser.BaseUnitSpecifierContext):
        pass

    def enterSIBaseUnitSpecifier(self, ctx: OpenSCENARIO2Parser.SIBaseUnitSpecifierContext):
        self.__ensure_scope(ctx)
        _ = self._define_and_enter(SiBaseExponentListScope(self.__current_scope), ctx)

    def exitSIBaseUnitSpecifier(self, ctx: OpenSCENARIO2Parser.SIBaseUnitSpecifierContext):
        self._leave_scope(ctx)

    def enterUnitDeclaration(self, ctx: OpenSCENARIO2Parser.UnitDeclarationContext):
        self.__ensure_scope(ctx)
        parent = self._push_parent(ctx)
        unit_name = ctx.Identifier().getText()

        physical_name = ctx.physicalTypeName().getText()
        if not self.__current_scope.resolve(physical_name):
            LOG_ERROR("PhysicalType: " + physical_name + " is not defined!", ctx.start)

        unit = self._define_or_reuse(
            self.__current_scope, unit_name,
            lambda: UnitSymbol(unit_name, self.__current_scope, physical_name),
            ctx, expected_cls=UnitSymbol
        )
        self.__current_scope = unit

        node = ast_node.UnitDeclaration(unit_name, physical_name)
        self._open_node(ctx, node, self.__current_scope)

    def exitUnitDeclaration(self, ctx: OpenSCENARIO2Parser.UnitDeclarationContext):
        self._pop_parent(ctx)
        self._leave_scope(ctx)

    def enterUnitSpecifier(self, ctx: OpenSCENARIO2Parser.UnitSpecifierContext):
        pass

    def exitUnitSpecifier(self, ctx: OpenSCENARIO2Parser.UnitSpecifierContext):
        pass

    def enterSIUnitSpecifier(self, ctx: OpenSCENARIO2Parser.SIUnitSpecifierContext):
        self.__ensure_scope(ctx)
        _ = self._define_and_enter(SiBaseExponentListScope(self.__current_scope), ctx)

    def exitSIUnitSpecifier(self, ctx: OpenSCENARIO2Parser.SIUnitSpecifierContext):
        self._leave_scope(ctx)

    def enterSIBaseExponentList(self, ctx: OpenSCENARIO2Parser.SIBaseExponentListContext):
        pass

    def exitSIBaseExponentList(self, ctx: OpenSCENARIO2Parser.SIBaseExponentListContext):
        pass

    def enterSIBaseExponent(self, ctx: OpenSCENARIO2Parser.SIBaseExponentContext):
        self.__ensure_scope(ctx)
        parent = self._push_parent(ctx)
        unit_name = ctx.Identifier().getText()
        value = ctx.integerLiteral().getText()

        # Guard duplicate exponents inside the same list
        existing = getattr(self.__current_scope, "symbols", {}).get(unit_name)
        if isinstance(existing, SiExpSymbol):
            try:
                existing.value = value
            except Exception:
                pass
            si_base_exponent = existing
        else:
            si_base_exponent = self.__current_scope.define(
                SiExpSymbol(unit_name, value, self.__current_scope), ctx.start
            ) or SiExpSymbol(unit_name, value, self.__current_scope)

        self.__current_scope = si_base_exponent

        node = ast_node.SIBaseExponent(unit_name)
        self._open_node(ctx, node, self.__current_scope)

    def exitSIBaseExponent(self, ctx: OpenSCENARIO2Parser.SIBaseExponentContext):
        self._pop_parent(ctx)
        self._leave_scope(ctx)

    def enterSIFactor(self, ctx: OpenSCENARIO2Parser.SIFactorContext):
        self.__ensure_scope(ctx)
        parent = self._push_parent(ctx)
        unit_name = "factor"

        factor_value = None
        if ctx.FloatLiteral():
            factor_value = ctx.FloatLiteral().getText()
        elif ctx.integerLiteral():
            factor_value = ctx.integerLiteral().getText()

        existing = getattr(self.__current_scope, "symbols", {}).get(unit_name)
        if existing is None:
            si_base_exponent = self.__current_scope.define(
                SiExpSymbol(unit_name, factor_value, self.__current_scope), ctx.start
            ) or SiExpSymbol(unit_name, factor_value, self.__current_scope)
            self.__current_scope = si_base_exponent
        else:
            try:
                existing.value = factor_value
            except Exception:
                pass
            self.__current_scope = existing

        node = ast_node.SIBaseExponent(unit_name)
        self._open_node(ctx, node, self.__current_scope)

        if ctx.FloatLiteral():
            saved = self._push_parent(ctx)
            float_value = ctx.FloatLiteral().getText()
            value_node = ast_node.FloatLiteral(float_value)
            self._open_node(ctx, value_node, self.__current_scope)
            self._pop_parent(ctx)

    def exitSIFactor(self, ctx: OpenSCENARIO2Parser.SIFactorContext):
        self._pop_parent(ctx)
        self._leave_scope(ctx)

    def enterSIOffset(self, ctx: OpenSCENARIO2Parser.SIOffsetContext):
        self.__ensure_scope(ctx)
        parent = self._push_parent(ctx)
        unit_name = "offset"

        offset_value = None
        if ctx.FloatLiteral():
            offset_value = ctx.FloatLiteral().getText()
        elif ctx.integerLiteral():
            offset_value = ctx.integerLiteral().getText()

        existing = getattr(self.__current_scope, "symbols", {}).get(unit_name)
        if existing is None:
            si_off = self.__current_scope.define(
                SiExpSymbol(unit_name, offset_value, self.__current_scope), ctx.start
            ) or SiExpSymbol(unit_name, offset_value, self.__current_scope)
            self.__current_scope = si_off
        else:
            try:
                existing.value = offset_value
            except Exception:
                pass
            self.__current_scope = existing

        node = ast_node.SIBaseExponent(unit_name)
        self._open_node(ctx, node, self.__current_scope)

        if ctx.FloatLiteral():
            saved = self._push_parent(ctx)
            value_node = ast_node.FloatLiteral(ctx.FloatLiteral().getText())
            self._open_node(ctx, value_node, self.__current_scope)
            self._pop_parent(ctx)

    def exitSIOffset(self, ctx: OpenSCENARIO2Parser.SIOffsetContext):
        self._pop_parent(ctx)
        self._leave_scope(ctx)

    # -------- Enums --------
    def enterEnumDeclaration(self, ctx: OpenSCENARIO2Parser.EnumDeclarationContext):
        parent = self._push_parent(ctx)
        enum_name = ctx.enumName().getText()

        enum = self._define_or_reuse(
            self.__current_scope, enum_name,
            lambda: EnumSymbol(enum_name, self.__current_scope),
            ctx, expected_cls=EnumSymbol
        )
        self.__current_scope = enum

        node = ast_node.EnumDeclaration(enum_name)
        self._open_node(ctx, node, self.__current_scope)

    def exitEnumDeclaration(self, ctx: OpenSCENARIO2Parser.EnumDeclarationContext):
        self._pop_parent(ctx)
        self._leave_scope(ctx)

    def enterEnumMemberDecl(self, ctx: OpenSCENARIO2Parser.EnumMemberDeclContext):
        parent = self._push_parent(ctx)
        member_name = ctx.enumMemberName().getText()

        member_value = None
        if ctx.enumMemberValue():
            if ctx.enumMemberValue().UintLiteral():
                member_value = int(ctx.enumMemberValue().UintLiteral().getText())
            elif ctx.enumMemberValue().HexUintLiteral():
                member_value = int(ctx.enumMemberValue().HexUintLiteral().getText(), 16)

        if member_value is None:
            member_value = self.__current_scope.last_index + 1

        # Guard duplicate members by name
        existing = getattr(self.__current_scope, "symbols", {}).get(member_name)
        if isinstance(existing, EnumMemberSymbol):
            # keep first; optionally update value if first had none
            pass
        else:
            enum_member = EnumMemberSymbol(member_name, self.__current_scope, member_value)
            self.__current_scope.define(enum_member, ctx.start)

        node = ast_node.EnumMemberDecl(member_name, member_value)
        self._open_node(ctx, node, self.__current_scope)

    def exitEnumMemberDecl(self, ctx: OpenSCENARIO2Parser.EnumMemberDeclContext):
        self._pop_parent(ctx)

    def enterEnumMemberValue(self, ctx: OpenSCENARIO2Parser.EnumMemberValueContext):
        pass

    def exitEnumMemberValue(self, ctx: OpenSCENARIO2Parser.EnumMemberValueContext):
        pass

    def enterEnumName(self, ctx: OpenSCENARIO2Parser.EnumNameContext):
        pass

    def exitEnumName(self, ctx: OpenSCENARIO2Parser.EnumNameContext):
        pass

    def enterEnumMemberName(self, ctx: OpenSCENARIO2Parser.EnumMemberNameContext):
        pass

    def exitEnumMemberName(self, ctx: OpenSCENARIO2Parser.EnumMemberNameContext):
        pass

    def enterEnumValueReference(self, ctx: OpenSCENARIO2Parser.EnumValueReferenceContext):
        parent = self._push_parent(ctx)
        enum_name = None
        if ctx.enumName():
            enum_name = ctx.enumName().getText()

        scope = self.__current_scope.resolve(enum_name)
        enum_member_name = ctx.enumMemberName().getText()

        if scope and isinstance(scope, EnumSymbol):
            if scope.symbols.get(enum_member_name):
                enum_value_reference = EnumValueRefSymbol(
                    enum_name,
                    enum_member_name,
                    scope.symbols[enum_member_name].elems_index,
                    scope,
                )
                # Value refs can repeat; keying is usually composite. Try/catch define.
                if not getattr(self.__current_scope, "symbols", {}).get(enum_member_name):
                    self.__current_scope.define(enum_value_reference, ctx.start)
            else:
                LOG_ERROR("Enum member " + enum_member_name + " not found!", ctx.start)
        else:
            LOG_ERROR("Enum " + enum_name + " not found!", ctx.start)

        node = ast_node.EnumValueReference(enum_name, enum_member_name)
        self._open_node(ctx, node, self.__current_scope)

    def exitEnumValueReference(self, ctx: OpenSCENARIO2Parser.EnumValueReferenceContext):
        self._pop_parent(ctx)

    # -------- Inherits condition --------
    def enterInheritsCondition(self, ctx: OpenSCENARIO2Parser.InheritsConditionContext):
        parent = self._push_parent(ctx)
        field_name = ctx.fieldName().getText()

        inherits_condition = self._define_or_reuse(
            self.__current_scope, field_name,
            lambda: InheritsConditionSymbol(field_name, self.__current_scope),
            ctx, expected_cls=InheritsConditionSymbol
        )
        self.__current_scope = inherits_condition

        bool_literal_node = None
        if ctx.BoolLiteral():
            bool_literal = ctx.BoolLiteral().getText()
            bool_literal_node = ast_node.BoolLiteral(bool_literal)
            bool_symbol = BoolSymbol(self.__current_scope, bool_literal)
            # allow multiple bools; no guard needed

            self.__current_scope.define(bool_symbol, ctx.start)

        node = ast_node.InheritsCondition(field_name, bool_literal_node)
        self._open_node(ctx, node, self.__current_scope)

    def exitInheritsCondition(self, ctx: OpenSCENARIO2Parser.InheritsConditionContext):
        self._pop_parent(ctx)
        self._leave_scope(ctx)

    # -------- Structs --------
    def enterStructDeclaration(self, ctx: OpenSCENARIO2Parser.StructDeclarationContext):
        parent = self._push_parent(ctx)
        struct_name = ctx.structName().getText()

        struct = self._define_or_reuse(
            self.__current_scope, struct_name,
            lambda: StructSymbol(struct_name, self.__current_scope),
            ctx, expected_cls=StructSymbol
        )
        self.__current_scope = struct

        node = ast_node.StructDeclaration(struct_name)
        self._open_node(ctx, node, self.__current_scope)

    def exitStructDeclaration(self, ctx: OpenSCENARIO2Parser.StructDeclarationContext):
        self._pop_parent(ctx)
        self._leave_scope(ctx)

    def enterStructInherts(self, ctx: OpenSCENARIO2Parser.StructInhertsContext):
        parent = self._push_parent(ctx)
        base_name = ctx.structName().getText()

        scope = None
        try:
            scope = self.__current_scope.resolve(base_name)
        except Exception:
            pass
        if scope is None:
            scope = self.__resolve_global_symbol(base_name)
        if scope is None:
            LOG_WARNING("inherits " + base_name + " is not resolved yet (deferring).", ctx.start)

        # allow multiple inherits references with same base (no collision under same key)
        struct_inherts = StructInhertsSymbol(base_name, self.__current_scope, scope)
        # safe-define under a composite key if available
        self.__current_scope.define(struct_inherts, ctx.start)

        node = ast_node.StructInherts(base_name)
        self._open_node(ctx, node, self.__current_scope)

    def exitStructInherts(self, ctx: OpenSCENARIO2Parser.StructInhertsContext):
        self._pop_parent(ctx)

    def enterStructMemberDecl(self, ctx: OpenSCENARIO2Parser.StructMemberDeclContext):
        pass

    def exitStructMemberDecl(self, ctx: OpenSCENARIO2Parser.StructMemberDeclContext):
        pass

    def enterFieldName(self, ctx: OpenSCENARIO2Parser.FieldNameContext):
        pass

    def exitFieldName(self, ctx: OpenSCENARIO2Parser.FieldNameContext):
        pass

    def enterStructName(self, ctx: OpenSCENARIO2Parser.StructNameContext):
        pass

    def exitStructName(self, ctx: OpenSCENARIO2Parser.StructNameContext):
        pass

    # -------- Actors / Actions --------
    def enterActorDeclaration(self, ctx: OpenSCENARIO2Parser.ActorDeclarationContext):
        parent = self._push_parent(ctx)
        actor_name = ctx.actorName().getText()

        actor_sym = self._define_or_reuse(
            self.__global_scope, actor_name,
            lambda: ActorSymbol(actor_name, self.__global_scope),
            ctx, expected_cls=ActorSymbol
        )

        self.__current_scope = actor_sym
        node = ast_node.ActorDeclaration(actor_name)
        self._open_node(ctx, node, self.__current_scope)

    def exitActorDeclaration(self, ctx: OpenSCENARIO2Parser.ActorDeclarationContext):
        self._pop_parent(ctx)
        self._leave_scope(ctx)

    def enterActionDeclaration(self, ctx: OpenSCENARIO2Parser.ActionDeclarationContext):
        parent = self._push_parent(ctx)

        actor_scope = None
        actor_name = None
        action_name = None

        if hasattr(ctx, "qualifiedBehaviorName") and ctx.qualifiedBehaviorName():
            qtxt = ctx.qualifiedBehaviorName().getText()
            a, b = self.__split_qualified(qtxt)
            actor_name, action_name = a, b

            if actor_name:
                actor_scope = self.__resolve_global_symbol(actor_name)
                if not isinstance(actor_scope, ActorSymbol):
                    LOG_ERROR(f"actorName: {actor_name} is not defined!", ctx.start)
                    actor_scope = self.__global_scope
            else:
                if isinstance(self.__current_scope, ActorSymbol):
                    actor_scope = self.__current_scope
                    actor_name = actor_scope.name
                else:
                    LOG_WARNING("Action without actor qualifier and not inside actor; placing under global.", ctx.start)
                    actor_scope = self.__global_scope
                    actor_name = "<?>"

        elif hasattr(ctx, "actorName") and ctx.actorName() and hasattr(ctx, "behaviorName") and ctx.behaviorName():
            actor_name = ctx.actorName().getText()
            action_name = ctx.behaviorName().getText()
            actor_scope = self.__resolve_global_symbol(actor_name)
            if not isinstance(actor_scope, ActorSymbol):
                LOG_ERROR(f"actorName: {actor_name} is not defined!", ctx.start)
                actor_scope = self.__global_scope

        else:
            if hasattr(ctx, "behaviorName") and ctx.behaviorName():
                action_name = ctx.behaviorName().getText()
            elif hasattr(ctx, "actionName") and ctx.actionName():
                action_name = ctx.actionName().getText()
            else:
                LOG_ERROR("action name missing", ctx.start)
                action_name = "<unnamed>"

            if isinstance(self.__current_scope, ActorSymbol):
                actor_scope = self.__current_scope
                actor_name = actor_scope.name
            else:
                LOG_WARNING("Action declared outside of an actor; using global.", ctx.start)
                actor_scope = self.__global_scope
                actor_name = "<?>"

        fq_name = f"{actor_name}.{action_name}"

        existing = None
        if isinstance(actor_scope, ActorSymbol) and hasattr(actor_scope, "symbols"):
            existing = actor_scope.symbols.get(action_name) or actor_scope.symbols.get(fq_name)

        if existing and isinstance(existing, ActionSymbol):
            action_sym = existing
        else:
            qsym = QualifiedBehaviorSymbol(fq_name, actor_scope)
            action_sym = ActionSymbol(qsym)
            action_sym = actor_scope.define(action_sym, ctx.start) or action_sym
            if hasattr(actor_scope, "symbols"):
                actor_scope.symbols.setdefault(action_name, action_sym)

        node = ast_node.ActionDeclaration(fq_name)
        self.__current_scope = action_sym
        self._open_node(ctx, node, self.__current_scope)
        self.__current_scope.declaration_address = node

    def exitActionDeclaration(self, ctx: OpenSCENARIO2Parser.ActionDeclarationContext):
        self._pop_parent(ctx)
        self._leave_scope(ctx)

    def enterActionInherts(self, ctx: OpenSCENARIO2Parser.ActionInhertsContext):
        parent = self._push_parent(ctx)
        qname = ctx.qualifiedBehaviorName().getText()

        target = self.__resolve_global(qname)
        if not (target and isinstance(target, ActionSymbol)):
            LOG_ERROR("inherits " + qname + " is not defined!", ctx.start)

        action_qsym = QualifiedBehaviorSymbol(qname, self.__current_scope)
        action_inherits = ActionInhertsSymbol(action_qsym, target)
        self.__current_scope.define(action_inherits, ctx.start)

        node = ast_node.ActionInherts(qname)
        self._open_node(ctx, node, self.__current_scope)

    def exitActionInherts(self, ctx: OpenSCENARIO2Parser.ActionInhertsContext):
        self._pop_parent(ctx)

    def enterActorMemberDecl(self, ctx: OpenSCENARIO2Parser.ActorMemberDeclContext):
        pass

    def exitActorMemberDecl(self, ctx: OpenSCENARIO2Parser.ActorMemberDeclContext):
        pass

    def enterActorName(self, ctx: OpenSCENARIO2Parser.ActorNameContext):
        pass

    def exitActorName(self, ctx: OpenSCENARIO2Parser.ActorNameContext):
        pass

    # -------- Scenarios --------
    def enterScenarioDeclaration(self, ctx: OpenSCENARIO2Parser.ScenarioDeclarationContext):
        parent = self._push_parent(ctx)
        qname = ctx.qualifiedBehaviorName().getText()

        actor_name, local = self.__split_qualified(qname)
        owner = self.__global_scope
        if actor_name:
            cand = self.__resolve_global_symbol(actor_name)
            if isinstance(cand, ActorSymbol):
                owner = cand

        # Guard scenario re-declare under same owner
        existing = getattr(owner, "symbols", {}).get(local) or getattr(owner, "symbols", {}).get(qname)
        if isinstance(existing, ScenarioSymbol):
            scenario = existing
        else:
            scenario_name = QualifiedBehaviorSymbol(qname, owner)
            scenario = ScenarioSymbol(scenario_name)
            scenario = owner.define(scenario, ctx.start) or scenario

        node = ast_node.ScenarioDeclaration(qname)
        self.__current_scope = scenario
        self._open_node(ctx, node, self.__current_scope)
        self.__current_scope.declaration_address = node

    def exitScenarioDeclaration(self, ctx: OpenSCENARIO2Parser.ScenarioDeclarationContext):
        self._pop_parent(ctx)
        self._leave_scope(ctx)

    def enterScenarioInherts(self, ctx: OpenSCENARIO2Parser.ScenarioInhertsContext):
        parent = self._push_parent(ctx)
        qualified_behavior_name = ctx.qualifiedBehaviorName().getText()

        scope = self.__current_scope.resolve(qualified_behavior_name)
        if scope is None:
            parts = qualified_behavior_name.split(".", 1)
            if len(parts) == 2:
                actor_name, scen_name = parts
                actor_scope = self.__current_scope.resolve(actor_name)
                if actor_scope and hasattr(actor_scope, "symbols"):
                    cand = actor_scope.symbols.get(scen_name)
                    if isinstance(cand, ScenarioSymbol):
                        scope = cand
                    else:
                        for sym in actor_scope.symbols.values():
                            if isinstance(sym, ScenarioSymbol):
                                qn = getattr(sym, "name", None)
                                beh = getattr(qn, "behavior", None)
                                if beh == scen_name:
                                    scope = sym
                                    break

        scenario_qname = QualifiedBehaviorSymbol(qualified_behavior_name, self.__current_scope)

        if scope is None or not isinstance(scope, ScenarioSymbol):
            LOG_ERROR("inherits " + qualified_behavior_name + " is not defined!", ctx.start)

        scenario_inherts = ScenarioInhertsSymbol(scenario_qname)
        scenario_inherts = self.__current_scope.define(scenario_inherts, ctx.start) or scenario_inherts
        self.__current_scope = scenario_inherts

        node = ast_node.ScenarioInherts(qualified_behavior_name)
        self._open_node(ctx, node, self.__current_scope)

    def exitScenarioInherts(self, ctx: OpenSCENARIO2Parser.ScenarioInhertsContext):
        self._pop_parent(ctx)
        self._leave_scope(ctx)

    def enterScenarioMemberDecl(self, ctx: OpenSCENARIO2Parser.ScenarioMemberDeclContext):
        parent = self._push_parent(ctx)

    def exitScenarioMemberDecl(self, ctx: OpenSCENARIO2Parser.ScenarioMemberDeclContext):
        self._pop_parent(ctx)

    def enterQualifiedBehaviorName(self, ctx: OpenSCENARIO2Parser.QualifiedBehaviorNameContext):
        pass

    def exitQualifiedBehaviorName(self, ctx: OpenSCENARIO2Parser.QualifiedBehaviorNameContext):
        pass

    def enterBehaviorName(self, ctx: OpenSCENARIO2Parser.BehaviorNameContext):
        pass

    def exitBehaviorName(self, ctx: OpenSCENARIO2Parser.BehaviorNameContext):
        pass

    # -------- Modifiers --------
    def enterModifierDeclaration(self, ctx: OpenSCENARIO2Parser.ModifierDeclarationContext):
        parent = self._push_parent(ctx)

        actor_name = ctx.actorName().getText() if ctx.actorName() else None
        owner = self._owner_for_actor_qualified(actor_name, ctx)
        modifier_name = ctx.modifierName().getText()

        modifier = self._define_or_reuse(
            owner, modifier_name,
            lambda: ModifierSymbol(modifier_name, owner),
            ctx, expected_cls=ModifierSymbol
        )
        self.__current_scope = modifier

        node = ast_node.ModifierDeclaration(actor_name, modifier_name)
        self._open_node(ctx, node, modifier)

    def exitModifierDeclaration(self, ctx: OpenSCENARIO2Parser.ModifierDeclarationContext):
        self._pop_parent(ctx)
        self._leave_scope(ctx)

    def enterModifierName(self, ctx: OpenSCENARIO2Parser.ModifierNameContext):
        pass

    def exitModifierName(self, ctx: OpenSCENARIO2Parser.ModifierNameContext):
        pass

    def enterTypeExtension(self, ctx: OpenSCENARIO2Parser.TypeExtensionContext):
        pass

    def exitTypeExtension(self, ctx: OpenSCENARIO2Parser.TypeExtensionContext):
        pass

    # -------- Enum / Structured type extensions --------
    def enterEnumTypeExtension(self, ctx: OpenSCENARIO2Parser.EnumTypeExtensionContext):
        parent = self._push_parent(ctx)
        enum_name = ctx.enumName().getText()

        if enum_name in self.__current_scope.symbols and isinstance(
            self.__current_scope.symbols[enum_name], EnumSymbol
        ):
            self.__current_scope = self.__current_scope.symbols[enum_name]
        else:
            LOG_ERROR(enum_name + " is not defined!", ctx.start)

        node = ast_node.EnumTypeExtension(enum_name)
        self._open_node(ctx, node, self.__current_scope)

    def exitEnumTypeExtension(self, ctx: OpenSCENARIO2Parser.EnumTypeExtensionContext):
        self._pop_parent(ctx)
        self._leave_scope(ctx)

    def enterStructuredTypeExtension(self, ctx: OpenSCENARIO2Parser.StructuredTypeExtensionContext):
        parent = self._push_parent(ctx)

        type_name = None
        if ctx.extendableTypeName().typeName():
            type_name = ctx.extendableTypeName().typeName().getText()
            if type_name in self.__current_scope.symbols:
                for extend_member in ctx.extensionMemberDecl():
                    if extend_member.structMemberDecl() and isinstance(
                        self.__current_scope.symbols[type_name], StructSymbol
                    ):
                        self.__current_scope = self.__current_scope.symbols[type_name]
                    elif extend_member.actorMemberDecl and isinstance(
                        self.__current_scope.symbols[type_name], ActorSymbol
                    ):
                        self.__current_scope = self.__current_scope.symbols[type_name]
                    elif extend_member.scenarioMemberDecl and isinstance(
                        self.__current_scope.symbols[type_name], ScenarioSymbol
                    ):
                        self.__current_scope = self.__current_scope.symbols[type_name]
                    elif extend_member.behaviorSpecification:
                        LOG_ERROR("I haven't written the code yet", ctx.start)
                    else:
                        LOG_ERROR(type_name + " is not defined!", ctx.start)
            else:
                LOG_ERROR(type_name + " is not defined!", ctx.start)

        qualified_behavior_name = None
        if ctx.extendableTypeName().qualifiedBehaviorName():
            qualified_behavior_name = ctx.extendableTypeName().qualifiedBehaviorName().getText()
            if qualified_behavior_name in self.__current_scope.symbols:
                for extend_member in ctx.extensionMemberDecl():
                    if extend_member.scenarioMemberDecl and isinstance(
                        self.__current_scope.symbols[qualified_behavior_name], ScenarioSymbol
                    ):
                        self.__current_scope = self.__current_scope.symbols[qualified_behavior_name]
                    elif extend_member.behaviorSpecification:
                        LOG_ERROR("not implemented", ctx.start)
                    else:
                        LOG_ERROR(qualified_behavior_name + " is not defined!", ctx.start)
            else:
                LOG_ERROR(qualified_behavior_name + " is Not defined!", ctx.start)

        node = ast_node.StructuredTypeExtension(type_name, qualified_behavior_name)
        self._open_node(ctx, node, self.__current_scope)

    def exitStructuredTypeExtension(self, ctx: OpenSCENARIO2Parser.StructuredTypeExtensionContext):
        self._pop_parent(ctx)
        self._leave_scope(ctx)

    def enterExtendableTypeName(self, ctx: OpenSCENARIO2Parser.ExtendableTypeNameContext):
        pass

    def exitExtendableTypeName(self, ctx: OpenSCENARIO2Parser.ExtendableTypeNameContext):
        pass

    def enterExtensionMemberDecl(self, ctx: OpenSCENARIO2Parser.ExtensionMemberDeclContext):
        pass

    def exitExtensionMemberDecl(self, ctx: OpenSCENARIO2Parser.ExtensionMemberDeclContext):
        pass

    # -------- Parameters (global) --------
    def enterGlobalParameterDeclaration(self, ctx: OpenSCENARIO2Parser.GlobalParameterDeclarationContext):
        defaultValue = None
        if ctx.defaultValue():
            defaultValue = ctx.defaultValue().getText()
        field_type = ctx.typeDeclarator().getText()
        parent = self._push_parent(ctx)

        field_name = []
        multi_field_name = ""
        for fn in ctx.fieldName():
            name = fn.Identifier().getText()
            if name in field_name:
                LOG_ERROR("Can not define same param in same scope!", ctx.start)
            field_name.append(name)
            multi_field_name = multi_field_name_append(multi_field_name, name)

        parameter = self._define_or_reuse(
            self.__current_scope, multi_field_name,
            lambda: ParameterSymbol(multi_field_name, self.__current_scope, field_type, defaultValue),
            ctx, expected_cls=ParameterSymbol
        )
        self.__current_scope = parameter

        node = ast_node.GlobalParameterDeclaration(field_name, field_type)
        self._open_node(ctx, node, self.__current_scope)

    def exitGlobalParameterDeclaration(self, ctx: OpenSCENARIO2Parser.GlobalParameterDeclarationContext):
        self._pop_parent(ctx)
        self._leave_scope(ctx)

    # -------- Types --------
    def enterTypeDeclarator(self, ctx: OpenSCENARIO2Parser.TypeDeclaratorContext):
        parent = self._push_parent(ctx)
        type_name = None
        if ctx.nonAggregateTypeDeclarator():
            type_name = ctx.nonAggregateTypeDeclarator().getText()
        elif ctx.aggregateTypeDeclarator():
            type_name = ctx.aggregateTypeDeclarator().getText()

        node = ast_node.Type(type_name)
        self._open_node(ctx, node, self.__current_scope)

    def exitTypeDeclarator(self, ctx: OpenSCENARIO2Parser.TypeDeclaratorContext):
        self._pop_parent(ctx)

    def enterNonAggregateTypeDeclarator(self, ctx: OpenSCENARIO2Parser.NonAggregateTypeDeclaratorContext):
        pass

    def exitNonAggregateTypeDeclarator(self, ctx: OpenSCENARIO2Parser.NonAggregateTypeDeclaratorContext):
        pass

    def enterAggregateTypeDeclarator(self, ctx: OpenSCENARIO2Parser.AggregateTypeDeclaratorContext):
        pass

    def exitAggregateTypeDeclarator(self, ctx: OpenSCENARIO2Parser.AggregateTypeDeclaratorContext):
        pass

    def enterListTypeDeclarator(self, ctx: OpenSCENARIO2Parser.ListTypeDeclaratorContext):
        pass

    def exitListTypeDeclarator(self, ctx: OpenSCENARIO2Parser.ListTypeDeclaratorContext):
        pass

    def enterPrimitiveType(self, ctx: OpenSCENARIO2Parser.PrimitiveTypeContext):
        pass

    def exitPrimitiveType(self, ctx: OpenSCENARIO2Parser.PrimitiveTypeContext):
        pass

    def enterTypeName(self, ctx: OpenSCENARIO2Parser.TypeNameContext):
        name = ctx.Identifier().getText()
        _ = self.__current_scope.resolve(name)
        pass

    def exitTypeName(self, ctx: OpenSCENARIO2Parser.TypeNameContext):
        pass

    # -------- Events --------
    def enterEventDeclaration(self, ctx: OpenSCENARIO2Parser.EventDeclarationContext):
        parent = self._push_parent(ctx)
        event_name = ctx.eventName().getText()

        event = self._define_or_reuse(
            self.__current_scope, event_name,
            lambda: EventSymbol(event_name, self.__current_scope),
            ctx, expected_cls=EventSymbol
        )
        self.__current_scope = event

        node = ast_node.EventDeclaration(event_name)
        self._open_node(ctx, node, self.__current_scope)
        self.__current_scope.declaration_address = node

    def exitEventDeclaration(self, ctx: OpenSCENARIO2Parser.EventDeclarationContext):
        self._pop_parent(ctx)
        self._leave_scope(ctx)

    def enterEventSpecification(self, ctx: OpenSCENARIO2Parser.EventSpecificationContext):
        pass

    def exitEventSpecification(self, ctx: OpenSCENARIO2Parser.EventSpecificationContext):
        pass

    def enterEventReference(self, ctx: OpenSCENARIO2Parser.EventReferenceContext):
        parent = self._push_parent(ctx)
        event_path = ctx.eventPath().getText()
        node = ast_node.EventReference(event_path)
        self._open_node(ctx, node, self.__current_scope)

    def exitEventReference(self, ctx: OpenSCENARIO2Parser.EventReferenceContext):
        self._pop_parent(ctx)

    def enterEventFieldDecl(self, ctx: OpenSCENARIO2Parser.EventFieldDeclContext):
        parent = self._push_parent(ctx)
        event_field_name = ctx.eventFieldName().getText()
        node = ast_node.EventFieldDecl(event_field_name)
        self._open_node(ctx, node, self.__current_scope)

    def exitEventFieldDecl(self, ctx: OpenSCENARIO2Parser.EventFieldDeclContext):
        self._pop_parent(ctx)

    def enterEventFieldName(self, ctx: OpenSCENARIO2Parser.EventFieldNameContext):
        pass

    def exitEventFieldName(self, ctx: OpenSCENARIO2Parser.EventFieldNameContext):
        pass

    def enterEventName(self, ctx: OpenSCENARIO2Parser.EventNameContext):
        pass

    def exitEventName(self, ctx: OpenSCENARIO2Parser.EventNameContext):
        pass

    def enterEventPath(self, ctx: OpenSCENARIO2Parser.EventPathContext):
        pass

    def exitEventPath(self, ctx: OpenSCENARIO2Parser.EventPathContext):
        pass

    def enterEventCondition(self, ctx: OpenSCENARIO2Parser.EventConditionContext):
        parent = self._push_parent(ctx)
        node = ast_node.EventCondition()
        self._open_node(ctx, node, self.__current_scope)

    def exitEventCondition(self, ctx: OpenSCENARIO2Parser.EventConditionContext):
        self._pop_parent(ctx)

    def enterRiseExpression(self, ctx: OpenSCENARIO2Parser.RiseExpressionContext):
        parent = self._push_parent(ctx)
        node = ast_node.RiseExpression()
        self._open_node(ctx, node, self.__current_scope)

    def exitRiseExpression(self, ctx: OpenSCENARIO2Parser.RiseExpressionContext):
        self._pop_parent(ctx)

    def enterFallExpression(self, ctx: OpenSCENARIO2Parser.FallExpressionContext):
        parent = self._push_parent(ctx)
        node = ast_node.FallExpression()
        self._open_node(ctx, node, self.__current_scope)

    def exitFallExpression(self, ctx: OpenSCENARIO2Parser.FallExpressionContext):
        self._pop_parent(ctx)

    def enterElapsedExpression(self, ctx: OpenSCENARIO2Parser.ElapsedExpressionContext):
        parent = self._push_parent(ctx)
        node = ast_node.ElapsedExpression()
        self._open_node(ctx, node, self.__current_scope)

    def exitElapsedExpression(self, ctx: OpenSCENARIO2Parser.ElapsedExpressionContext):
        self._pop_parent(ctx)

    def enterEveryExpression(self, ctx: OpenSCENARIO2Parser.EveryExpressionContext):
        parent = self._push_parent(ctx)
        node = ast_node.EveryExpression()
        self._open_node(ctx, node, self.__current_scope)

    def exitEveryExpression(self, ctx: OpenSCENARIO2Parser.EveryExpressionContext):
        self._pop_parent(ctx)

    def enterBoolExpression(self, ctx: OpenSCENARIO2Parser.BoolExpressionContext):
        pass

    def exitBoolExpression(self, ctx: OpenSCENARIO2Parser.BoolExpressionContext):
        pass

    def enterDurationExpression(self, ctx: OpenSCENARIO2Parser.DurationExpressionContext):
        pass

    def exitDurationExpression(self, ctx: OpenSCENARIO2Parser.DurationExpressionContext):
        pass

    # -------- Fields / Variables / Parameters --------
    def enterFieldDeclaration(self, ctx: OpenSCENARIO2Parser.FieldDeclarationContext):
        pass

    def exitFieldDeclaration(self, ctx: OpenSCENARIO2Parser.FieldDeclarationContext):
        pass

    def enterParameterDeclaration(self, ctx: OpenSCENARIO2Parser.ParameterDeclarationContext):
        self.__ensure_scope(ctx)
        defaultValue = None
        if ctx.defaultValue():
            defaultValue = ctx.defaultValue().getText()
        field_type = ctx.typeDeclarator().getText()
        parent = self._push_parent(ctx)
        field_name = []

        multi_field_name = ""
        for fn in ctx.fieldName():
            name = fn.Identifier().getText()
            if name in field_name:
                LOG_ERROR("Can not define same param in same scope!", ctx.start)
            field_name.append(name)
            multi_field_name = multi_field_name_append(multi_field_name, name)

        parameter = self._define_or_reuse(
            self.__current_scope, multi_field_name,
            lambda: ParameterSymbol(multi_field_name, self.__current_scope, field_type, defaultValue),
            ctx, expected_cls=ParameterSymbol
        )
        self.__current_scope = parameter

        node = ast_node.ParameterDeclaration(field_name, field_type)
        self._open_node(ctx, node, self.__current_scope)

    def exitParameterDeclaration(self, ctx: OpenSCENARIO2Parser.ParameterDeclarationContext):
        self._pop_parent(ctx)
        self._leave_scope(ctx)

    def enterVariableDeclaration(self, ctx: OpenSCENARIO2Parser.VariableDeclarationContext):
        self.__ensure_scope(ctx)
        parent = self._push_parent(ctx)
        field_name = []
        defaultValue = None
        if ctx.sampleExpression():
            defaultValue = ctx.sampleExpression().getText()
        elif ctx.valueExp():
            defaultValue = ctx.valueExp().getText()

        multi_field_name = ""
        for fn in ctx.fieldName():
            name = fn.Identifier().getText()
            if name in field_name:
                LOG_ERROR("Can not define same param in same scope!", ctx.start)
            field_name.append(name)
            multi_field_name = multi_field_name_append(multi_field_name, name)

        field_type = ctx.typeDeclarator().getText()

        variable = self._define_or_reuse(
            self.__current_scope, multi_field_name,
            lambda: VariableSymbol(multi_field_name, self.__current_scope, field_type, defaultValue),
            ctx, expected_cls=VariableSymbol
        )
        self.__current_scope = variable

        node = ast_node.VariableDeclaration(field_name, field_type)
        self._open_node(ctx, node, self.__current_scope)

    def exitVariableDeclaration(self, ctx: OpenSCENARIO2Parser.VariableDeclarationContext):
        self._pop_parent(ctx)
        self._leave_scope(ctx)

    def enterSampleExpression(self, ctx: OpenSCENARIO2Parser.SampleExpressionContext):
        parent = self._push_parent(ctx)
        node = ast_node.SampleExpression()
        self._open_node(ctx, node, self.__current_scope)

    def exitSampleExpression(self, ctx: OpenSCENARIO2Parser.SampleExpressionContext):
        self._pop_parent(ctx)

    def enterDefaultValue(self, ctx: OpenSCENARIO2Parser.DefaultValueContext):
        pass

    def exitDefaultValue(self, ctx: OpenSCENARIO2Parser.DefaultValueContext):
        pass

    def enterParameterWithDeclaration(self, ctx: OpenSCENARIO2Parser.ParameterWithDeclarationContext):
        pass

    def exitParameterWithDeclaration(self, ctx: OpenSCENARIO2Parser.ParameterWithDeclarationContext):
        pass

    def enterParameterWithMember(self, ctx: OpenSCENARIO2Parser.ParameterWithMemberContext):
        pass

    def exitParameterWithMember(self, ctx: OpenSCENARIO2Parser.ParameterWithMemberContext):
        pass

    # -------- Constraints --------
    def enterConstraintDeclaration(self, ctx: OpenSCENARIO2Parser.ConstraintDeclarationContext):
        pass

    def exitConstraintDeclaration(self, ctx: OpenSCENARIO2Parser.ConstraintDeclarationContext):
        pass

    def enterKeepConstraintDeclaration(self, ctx: OpenSCENARIO2Parser.KeepConstraintDeclarationContext):
        parent = self._push_parent(ctx)
        constraint_qualifier = None
        if ctx.constraintQualifier():
            constraint_qualifier = ctx.constraintQualifier().getText()

        # Keep scopes can repeat with same qualifier; reuse.
        keep_symbol = self._define_or_reuse(
            self.__current_scope, constraint_qualifier or "__keep__",
            lambda: KeepScope(self.__current_scope, constraint_qualifier),
            ctx, expected_cls=KeepScope
        )
        self.__current_scope = keep_symbol

        node = ast_node.KeepConstraintDeclaration(constraint_qualifier)
        self._open_node(ctx, node, self.__current_scope)

    def exitKeepConstraintDeclaration(self, ctx: OpenSCENARIO2Parser.KeepConstraintDeclarationContext):
        self._pop_parent(ctx)
        self._leave_scope(ctx)

    def enterConstraintQualifier(self, ctx: OpenSCENARIO2Parser.ConstraintQualifierContext):
        pass

    def exitConstraintQualifier(self, ctx: OpenSCENARIO2Parser.ConstraintQualifierContext):
        pass

    def enterConstraintExpression(self, ctx: OpenSCENARIO2Parser.ConstraintExpressionContext):
        pass

    def exitConstraintExpression(self, ctx: OpenSCENARIO2Parser.ConstraintExpressionContext):
        pass

    def enterRemoveDefaultDeclaration(self, ctx: OpenSCENARIO2Parser.RemoveDefaultDeclarationContext):
        parent = self._push_parent(ctx)
        node = ast_node.RemoveDefaultDeclaration()
        self._open_node(ctx, node, self.__current_scope)

    def exitRemoveDefaultDeclaration(self, ctx: OpenSCENARIO2Parser.RemoveDefaultDeclarationContext):
        self._pop_parent(ctx)

    def enterParameterReference(self, ctx: OpenSCENARIO2Parser.ParameterReferenceContext):
        parent = self._push_parent(ctx)
        field_name = None
        if ctx.fieldName():
            field_name = ctx.fieldName().getText()

        field_access = None
        if ctx.fieldAccess():
            field_access = ctx.fieldAccess().getText()

        node = ast_node.ParameterReference(field_name, field_access)
        self._open_node(ctx, node, self.__current_scope)

    def exitParameterReference(self, ctx: OpenSCENARIO2Parser.ParameterReferenceContext):
        self._pop_parent(ctx)

    def enterModifierInvocation(self, ctx: OpenSCENARIO2Parser.ModifierInvocationContext):
        parent = self._push_parent(ctx)
        modifier_name = ctx.modifierName().getText()

        actor = None
        if ctx.actorExpression():
            actor = ctx.actorExpression().getText()

        if ctx.behaviorExpression():
            actor = ctx.behaviorExpression().getText()

        scope = self.__current_scope
        if actor is not None:
            resolved = self.__current_scope.resolve(actor)
            if resolved:
                scope = resolved

        node = ast_node.ModifierInvocation(actor, modifier_name)
        self._open_node(ctx, node, scope)

    def exitModifierInvocation(self, ctx: OpenSCENARIO2Parser.ModifierInvocationContext):
        self._pop_parent(ctx)

    def enterBehaviorExpression(self, ctx: OpenSCENARIO2Parser.BehaviorExpressionContext):
        pass

    def exitBehaviorExpression(self, ctx: OpenSCENARIO2Parser.BehaviorExpressionContext):
        pass

    def enterBehaviorSpecification(self, ctx: OpenSCENARIO2Parser.BehaviorSpecificationContext):
        pass

    def exitBehaviorSpecification(self, ctx: OpenSCENARIO2Parser.BehaviorSpecificationContext):
        pass

    def enterOnDirective(self, ctx: OpenSCENARIO2Parser.OnDirectiveContext):
        parent = self._push_parent(ctx)
        node = ast_node.OnDirective()
        self._open_node(ctx, node, self.__current_scope)

    def exitOnDirective(self, ctx: OpenSCENARIO2Parser.OnDirectiveContext):
        self._pop_parent(ctx)

    def enterOnMember(self, ctx: OpenSCENARIO2Parser.OnMemberContext):
        pass

    def exitOnMember(self, ctx: OpenSCENARIO2Parser.OnMemberContext):
        pass

    def enterDoDirective(self, ctx: OpenSCENARIO2Parser.DoDirectiveContext):
        parent = self._push_parent(ctx)

        do_directive_scope = self._define_or_reuse(
            self.__current_scope, "__do__",
            lambda: DoDirectiveScope(self.__current_scope),
            ctx, expected_cls=DoDirectiveScope
        )
        self.__current_scope = do_directive_scope

        node = ast_node.DoDirective()
        self._open_node(ctx, node, self.__current_scope)

    def exitDoDirective(self, ctx: OpenSCENARIO2Parser.DoDirectiveContext):
        self._pop_parent(ctx)
        self._leave_scope(ctx)

    def enterDoMember(self, ctx: OpenSCENARIO2Parser.DoMemberContext):
        parent = self._push_parent(ctx)
        label_name = None

        if ctx.labelName():
            label_name = ctx.labelName().getText()

        composition_operator = None
        if ctx.composition():
            composition_operator = ctx.composition().compositionOperator().getText()

        if composition_operator is not None:
            key = f"{label_name or ''}@{composition_operator}"
            domember = self._define_or_reuse(
                self.__current_scope, key,
                lambda: DoMemberSymbol(label_name, self.__current_scope, composition_operator),
                ctx, expected_cls=DoMemberSymbol
            )
            self.__current_scope = domember

            node = ast_node.DoMember(label_name, composition_operator)
            self._open_node(ctx, node, self.__current_scope)

    def exitDoMember(self, ctx: OpenSCENARIO2Parser.DoMemberContext):
        self._pop_parent(ctx)
        if ctx.composition() is not None:
            self._leave_scope(ctx)

    def enterComposition(self, ctx: OpenSCENARIO2Parser.CompositionContext):
        pass

    def exitComposition(self, ctx: OpenSCENARIO2Parser.CompositionContext):
        pass

    def enterCompositionOperator(self, ctx: OpenSCENARIO2Parser.CompositionOperatorContext):
        pass

    def exitCompositionOperator(self, ctx: OpenSCENARIO2Parser.CompositionOperatorContext):
        pass

    # -------- Behavior invocation / etc. --------
    def enterBehaviorInvocation(self, ctx: OpenSCENARIO2Parser.BehaviorInvocationContext):
        parent = self._push_parent(ctx)
        actor = None
        behavior_name = ctx.behaviorName().getText()

        if ctx.actorExpression():
            actor = ctx.actorExpression().getText()
            fq_name = f"{actor}.{behavior_name}"
        else:
            fq_name = behavior_name

        scope = self.__resolve_global(fq_name)
        if scope is None:
            try:
                scope = self.__current_scope.resolve(fq_name)
            except Exception:
                scope = None

        node = ast_node.BehaviorInvocation(actor, behavior_name)
        self._open_node(ctx, node, scope)

    def exitBehaviorInvocation(self, ctx: OpenSCENARIO2Parser.BehaviorInvocationContext):
        self._pop_parent(ctx)

    def enterBehaviorWithDeclaration(self, ctx: OpenSCENARIO2Parser.BehaviorWithDeclarationContext):
        pass

    def exitBehaviorWithDeclaration(self, ctx: OpenSCENARIO2Parser.BehaviorWithDeclarationContext):
        pass

    def enterBehaviorWithMember(self, ctx: OpenSCENARIO2Parser.BehaviorWithMemberContext):
        parent = self._push_parent(ctx)

    def exitBehaviorWithMember(self, ctx: OpenSCENARIO2Parser.BehaviorWithMemberContext):
        self._pop_parent(ctx)

    def enterLabelName(self, ctx: OpenSCENARIO2Parser.LabelNameContext):
        pass

    def exitLabelName(self, ctx: OpenSCENARIO2Parser.LabelNameContext):
        pass

    def enterActorExpression(self, ctx: OpenSCENARIO2Parser.ActorExpressionContext):
        pass

    def exitActorExpression(self, ctx: OpenSCENARIO2Parser.ActorExpressionContext):
        pass

    # -------- Wait / Emit / Call / Until --------
    def enterWaitDirective(self, ctx: OpenSCENARIO2Parser.WaitDirectiveContext):
        parent = self._push_parent(ctx)

        wait_scope = self._define_or_reuse(
            self.__current_scope, "__wait__",
            lambda: WaitSymbol(self.__current_scope),
            ctx, expected_cls=WaitSymbol
        )
        self.__current_scope = wait_scope

        node = ast_node.WaitDirective()
        self._open_node(ctx, node, self.__current_scope)

    def exitWaitDirective(self, ctx: OpenSCENARIO2Parser.WaitDirectiveContext):
        self._pop_parent(ctx)
        self._leave_scope(ctx)

    def enterEmitDirective(self, ctx: OpenSCENARIO2Parser.EmitDirectiveContext):
        parent = self._push_parent(ctx)
        event_name = ctx.eventName().getText()
        node = ast_node.EmitDirective(event_name)
        self._open_node(ctx, node, self.__current_scope)

    def exitEmitDirective(self, ctx: OpenSCENARIO2Parser.EmitDirectiveContext):
        self._pop_parent(ctx)

    def enterCallDirective(self, ctx: OpenSCENARIO2Parser.CallDirectiveContext):
        parent = self._push_parent(ctx)
        method_name = ctx.methodInvocation().postfixExp().getText()

        node = ast_node.CallDirective(method_name)
        self._open_node(ctx, node, self.__current_scope)

    def exitCallDirective(self, ctx: OpenSCENARIO2Parser.CallDirectiveContext):
        self._pop_parent(ctx)

    def enterUntilDirective(self, ctx: OpenSCENARIO2Parser.UntilDirectiveContext):
        parent = self._push_parent(ctx)
        node = ast_node.UntilDirective()
        self._open_node(ctx, node, self.__current_scope)

    def exitUntilDirective(self, ctx: OpenSCENARIO2Parser.UntilDirectiveContext):
        self._pop_parent(ctx)

    def enterMethodInvocation(self, ctx: OpenSCENARIO2Parser.MethodInvocationContext):
        pass

    def exitMethodInvocation(self, ctx: OpenSCENARIO2Parser.MethodInvocationContext):
        pass

    def enterMethodDeclaration(self, ctx: OpenSCENARIO2Parser.MethodDeclarationContext):
        parent = self._push_parent(ctx)
        method_name = ctx.methodName().getText()
        return_type = None
        if ctx.returnType():
            return_type = ctx.returnType().getText()

        method = self._define_or_reuse(
            self.__current_scope, method_name,
            lambda: MethodSymbol(method_name, self.__current_scope),
            ctx, expected_cls=MethodSymbol
        )
        self.__current_scope = method

        node = ast_node.MethodDeclaration(method_name, return_type)
        self._open_node(ctx, node, self.__current_scope)
        self.__current_scope.declaration_address = node

    def exitMethodDeclaration(self, ctx: OpenSCENARIO2Parser.MethodDeclarationContext):
        self._pop_parent(ctx)
        self._leave_scope(ctx)

    def enterReturnType(self, ctx: OpenSCENARIO2Parser.ReturnTypeContext):
        pass

    def exitReturnType(self, ctx: OpenSCENARIO2Parser.ReturnTypeContext):
        pass

    def enterMethodImplementation(self, ctx: OpenSCENARIO2Parser.MethodImplementationContext):
        parent = self._push_parent(ctx)
        qualifier = None
        if ctx.methodQualifier():
            qualifier = ctx.methodQualifier().getText()

        if ctx.expression():
            _type = "expression"
        elif ctx.structuredIdentifier():
            _type = "external"
        else:
            _type = "undefined"

        external_name = None
        if ctx.structuredIdentifier():
            external_name = ctx.structuredIdentifier().getText()

        node = ast_node.MethodBody(qualifier, _type, external_name)
        self._open_node(ctx, node, self.__current_scope)

    def exitMethodImplementation(self, ctx: OpenSCENARIO2Parser.MethodImplementationContext):
        self._pop_parent(ctx)

    def enterMethodQualifier(self, ctx: OpenSCENARIO2Parser.MethodQualifierContext):
        pass

    def exitMethodQualifier(self, ctx: OpenSCENARIO2Parser.MethodQualifierContext):
        pass

    def enterMethodName(self, ctx: OpenSCENARIO2Parser.MethodNameContext):
        pass

    def exitMethodName(self, ctx: OpenSCENARIO2Parser.MethodNameContext):
        pass

    # -------- Coverage --------
    def enterCoverageDeclaration(self, ctx: OpenSCENARIO2Parser.CoverageDeclarationContext):
        pass

    def exitCoverageDeclaration(self, ctx: OpenSCENARIO2Parser.CoverageDeclarationContext):
        pass

    def enterCoverDeclaration(self, ctx: OpenSCENARIO2Parser.CoverageDeclarationContext):
        parent = self._push_parent(ctx)
        target_name = None
        if ctx.targetName():
            target_name = ctx.targetName().getText()

        node = ast_node.coverDeclaration(target_name)
        self._open_node(ctx, node, self.__current_scope)

    def exitCoverDeclaration(self, ctx: OpenSCENARIO2Parser.CoverageDeclarationContext):
        self._pop_parent(ctx)

    def enterRecordDeclaration(self, ctx: OpenSCENARIO2Parser.RecordDeclarationContext):
        parent = self._push_parent(ctx)
        target_name = None
        if ctx.targetName():
            target_name = ctx.targetName().getText()

        node = ast_node.recordDeclaration(target_name)
        self._open_node(ctx, node, self.__current_scope)

    def exitRecordDeclaration(self, ctx: OpenSCENARIO2Parser.RecordDeclarationContext):
        self._pop_parent(ctx)

    def enterCoverageExpression(self, ctx: OpenSCENARIO2Parser.CoverageExpressionContext):
        parent = self._push_parent(ctx)
        argument_name = "expression"
        node = ast_node.NamedArgument(argument_name)
        self._open_node(ctx, node, self.__current_scope)

    def exitCoverageExpression(self, ctx: OpenSCENARIO2Parser.CoverageExpressionContext):
        self._pop_parent(ctx)

    def enterCoverageUnit(self, ctx: OpenSCENARIO2Parser.CoverageUnitContext):
        parent = self._push_parent(ctx)
        argument_name = "unit"
        node = ast_node.NamedArgument(argument_name)
        self._open_node(ctx, node, self.__current_scope)

        saved = self._push_parent(ctx)  # parent is NamedArgument
        unit_ident = ast_node.Identifier(ctx.Identifier().getText())
        self._open_node(ctx, unit_ident, self.__current_scope)
        self._pop_parent(ctx)

    def exitCoverageUnit(self, ctx: OpenSCENARIO2Parser.CoverageUnitContext):
        self._pop_parent(ctx)

    def enterCoverageRange(self, ctx: OpenSCENARIO2Parser.CoverageRangeContext):
        parent = self._push_parent(ctx)
        argument_name = "range"
        node = ast_node.NamedArgument(argument_name)
        self._open_node(ctx, node, self.__current_scope)

    def exitCoverageRange(self, ctx: OpenSCENARIO2Parser.CoverageRangeContext):
        self._pop_parent(ctx)

    def enterCoverageEvery(self, ctx: OpenSCENARIO2Parser.CoverageEveryContext):
        parent = self._push_parent(ctx)
        argument_name = "every"
        node = ast_node.NamedArgument(argument_name)
        self._open_node(ctx, node, self.__current_scope)

    def exitCoverageEvery(self, ctx: OpenSCENARIO2Parser.CoverageEveryContext):
        self._pop_parent(ctx)

    def enterCoverageEvent(self, ctx: OpenSCENARIO2Parser.CoverageEventContext):
        parent = self._push_parent(ctx)
        argument_name = "event"
        node = ast_node.NamedArgument(argument_name)
        self._open_node(ctx, node, self.__current_scope)

    def exitCoverageEvent(self, ctx: OpenSCENARIO2Parser.CoverageEventContext):
        self._pop_parent(ctx)

    def enterCoverageNameArgument(self, ctx: OpenSCENARIO2Parser.CoverageNameArgumentContext):
        pass

    def exitCoverageNameArgument(self, ctx: OpenSCENARIO2Parser.CoverageNameArgumentContext):
        pass

    def enterTargetName(self, ctx: OpenSCENARIO2Parser.TargetNameContext):
        pass

    def exitTargetName(self, ctx: OpenSCENARIO2Parser.TargetNameContext):
        pass

    # -------- Expressions --------
    def enterExpression(self, ctx: OpenSCENARIO2Parser.ExpressionContext):
        pass

    def exitExpression(self, ctx: OpenSCENARIO2Parser.ExpressionContext):
        pass

    def enterTernaryOpExp(self, ctx: OpenSCENARIO2Parser.TernaryOpExpContext):
        pass

    def exitTernaryOpExp(self, ctx: OpenSCENARIO2Parser.TernaryOpExpContext):
        pass

    def enterImplication(self, ctx: OpenSCENARIO2Parser.ImplicationContext):
        parent = self._push_parent(ctx)
        if len(ctx.disjunction()) > 1:
            operator = "=>"
            node = ast_node.LogicalExpression(operator)
            self._open_node(ctx, node, self.__current_scope)

    def exitImplication(self, ctx: OpenSCENARIO2Parser.ImplicationContext):
        self._pop_parent(ctx)

    def enterDisjunction(self, ctx: OpenSCENARIO2Parser.DisjunctionContext):
        parent = self._push_parent(ctx)
        if len(ctx.conjunction()) > 1:
            operator = "or"
            node = ast_node.LogicalExpression(operator)
            self._open_node(ctx, node, self.__current_scope)

    def exitDisjunction(self, ctx: OpenSCENARIO2Parser.DisjunctionContext):
        self._pop_parent(ctx)

    def enterConjunction(self, ctx: OpenSCENARIO2Parser.ConjunctionContext):
        parent = self._push_parent(ctx)
        if len(ctx.inversion()) > 1:
            operator = "and"
            node = ast_node.LogicalExpression(operator)
            self._open_node(ctx, node, self.__current_scope)

    def exitConjunction(self, ctx: OpenSCENARIO2Parser.ConjunctionContext):
        self._pop_parent(ctx)

    def enterInversion(self, ctx: OpenSCENARIO2Parser.InversionContext):
        parent = self._push_parent(ctx)
        if ctx.relation() is None:
            operator = "not"
            node = ast_node.LogicalExpression(operator)
            self._open_node(ctx, node, self.__current_scope)

    def exitInversion(self, ctx: OpenSCENARIO2Parser.InversionContext):
        self._pop_parent(ctx)

    def enterRelation(self, ctx: OpenSCENARIO2Parser.RelationContext):
        parent = self._push_parent(ctx)

    def exitRelation(self, ctx: OpenSCENARIO2Parser.RelationContext):
        self._pop_parent(ctx)

    def enterRelationExp(self, ctx: OpenSCENARIO2Parser.RelationExpContext):
        parent = self._push_parent(ctx)
        operator = ctx.relationalOp().getText()
        node = ast_node.RelationExpression(operator)
        self._open_node(ctx, node, self.__current_scope)

    def exitRelationExp(self, ctx: OpenSCENARIO2Parser.RelationExpContext):
        self._pop_parent(ctx)

    def enterRelationalOp(self, ctx: OpenSCENARIO2Parser.RelationalOpContext):
        pass

    def exitRelationalOp(self, ctx: OpenSCENARIO2Parser.RelationalOpContext):
        pass

    def enterSum(self, ctx: OpenSCENARIO2Parser.SumContext):
        pass

    def exitSum(self, ctx: OpenSCENARIO2Parser.SumContext):
        pass

    def enterAdditiveExp(self, ctx: OpenSCENARIO2Parser.AdditiveExpContext):
        parent = self._push_parent(ctx)
        operator = ctx.additiveOp().getText()
        node = ast_node.BinaryExpression(operator)
        self._open_node(ctx, node, self.__current_scope)

    def exitAdditiveExp(self, ctx: OpenSCENARIO2Parser.AdditiveExpContext):
        self._pop_parent(ctx)

    def enterAdditiveOp(self, ctx: OpenSCENARIO2Parser.AdditiveOpContext):
        pass

    def exitAdditiveOp(self, ctx: OpenSCENARIO2Parser.AdditiveOpContext):
        pass

    def enterMultiplicativeExp(self, ctx: OpenSCENARIO2Parser.MultiplicativeExpContext):
        parent = self._push_parent(ctx)
        operator = ctx.multiplicativeOp().getText()
        node = ast_node.BinaryExpression(operator)
        self._open_node(ctx, node, self.__current_scope)

    def exitMultiplicativeExp(self, ctx: OpenSCENARIO2Parser.MultiplicativeExpContext):
        self._pop_parent(ctx)

    def enterTerm(self, ctx: OpenSCENARIO2Parser.TermContext):
        pass

    def exitTerm(self, ctx: OpenSCENARIO2Parser.TermContext):
        pass

    def enterMultiplicativeOp(self, ctx: OpenSCENARIO2Parser.MultiplicativeOpContext):
        pass

    def exitMultiplicativeOp(self, ctx: OpenSCENARIO2Parser.MultiplicativeOpContext):
        pass

    def enterFactor(self, ctx: OpenSCENARIO2Parser.FactorContext):
        pass

    def exitFactor(self, ctx: OpenSCENARIO2Parser.FactorContext):
        pass

    def enterPrimaryExpression(self, ctx: OpenSCENARIO2Parser.PrimaryExpressionContext):
        pass

    def exitPrimaryExpression(self, ctx: OpenSCENARIO2Parser.PrimaryExpressionContext):
        pass

    def enterCastExpression(self, ctx: OpenSCENARIO2Parser.CastExpressionContext):
        parent = self._push_parent(ctx)
        object = ctx.postfixExp().getText()
        target_type = ctx.typeDeclarator().getText()
        node = ast_node.CastExpression(object, target_type)
        self._open_node(ctx, node, self.__current_scope)

    def exitCastExpression(self, ctx: OpenSCENARIO2Parser.CastExpressionContext):
        self._pop_parent(ctx)

    def enterFunctionApplicationExpression(self, ctx: OpenSCENARIO2Parser.FunctionApplicationExpressionContext):
        parent = self._push_parent(ctx)
        func_name = ctx.postfixExp().getText()
        scope = self.__current_scope
        func_name_list = func_name.split(".", 1)
        if len(func_name_list) > 1:
            scope = scope.resolve(func_name_list[0])
        node = ast_node.FunctionApplicationExpression(func_name)
        self._open_node(ctx, node, scope)

    def exitFunctionApplicationExpression(self, ctx: OpenSCENARIO2Parser.FunctionApplicationExpressionContext):
        self._pop_parent(ctx)

    def enterFieldAccessExpression(self, ctx: OpenSCENARIO2Parser.FieldAccessExpressionContext):
        parent = self._push_parent(ctx)
        field_name = ctx.postfixExp().getText() + "." + ctx.fieldName().getText()

        if ctx.postfixExp().getText() == "it":
            field_name = self.__current_scope.get_enclosing_scope().type
            scope = None
            if self.__current_scope.resolve(field_name):
                scope = self.__current_scope.resolve(field_name)
                if ctx.fieldName().getText() in scope.symbols:
                    pass
                else:
                    LOG_ERROR(
                        ctx.fieldName().getText()
                        + " is not defined in scope: "
                        + self.__current_scope.get_enclosing_scope().type,
                        ctx.start,
                    )
            else:
                LOG_ERROR("it -> " + field_name + " is not defined!", ctx.start)

        node = ast_node.FieldAccessExpression(field_name)
        self._open_node(ctx, node, self.__current_scope)

    def exitFieldAccessExpression(self, ctx: OpenSCENARIO2Parser.FieldAccessExpressionContext):
        self._pop_parent(ctx)

    def enterElementAccessExpression(self, ctx: OpenSCENARIO2Parser.ElementAccessExpressionContext):
        parent = self._push_parent(ctx)
        list_name = ctx.postfixExp().getText()
        index = ctx.expression().getText()
        node = ast_node.ElementAccessExpression(list_name, index)
        self._open_node(ctx, node, self.__current_scope)

    def exitElementAccessExpression(self, ctx: OpenSCENARIO2Parser.ElementAccessExpressionContext):
        self._pop_parent(ctx)

    def enterTypeTestExpression(self, ctx: OpenSCENARIO2Parser.TypeTestExpressionContext):
        parent = self._push_parent(ctx)
        object = ctx.postfixExp().getText()
        target_type = ctx.typeDeclarator().getText()
        node = ast_node.TypeTestExpression(object, target_type)
        self._open_node(ctx, node, self.__current_scope)

    def exitTypeTestExpression(self, ctx: OpenSCENARIO2Parser.TypeTestExpressionContext):
        self._pop_parent(ctx)

    def enterFieldAccess(self, ctx: OpenSCENARIO2Parser.FieldAccessContext):
        parent = self._push_parent(ctx)
        field_name = ctx.postfixExp().getText() + "." + ctx.fieldName().getText()
        node = ast_node.FieldAccessExpression(field_name)
        self._open_node(ctx, node, self.__current_scope)

    def exitFieldAccess(self, ctx: OpenSCENARIO2Parser.FieldAccessContext):
        self._pop_parent(ctx)

    def enterPrimaryExp(self, ctx: OpenSCENARIO2Parser.PrimaryExpContext):
        pass

    def exitPrimaryExp(self, ctx: OpenSCENARIO2Parser.PrimaryExpContext):
        pass

    def enterValueExp(self, ctx: OpenSCENARIO2Parser.ValueExpContext):
        parent = self._push_parent(ctx)
        value = None
        node = None
        if ctx.FloatLiteral():
            value = ctx.FloatLiteral().getText()
            node = ast_node.FloatLiteral(value)
        elif ctx.BoolLiteral():
            value = ctx.BoolLiteral().getText()
            node = ast_node.BoolLiteral(value)
        elif ctx.StringLiteral():
            value = ctx.StringLiteral().getText()
            value = value.strip('"')
            node = ast_node.StringLiteral(value)

        if node is not None:
            self._open_node(ctx, node, self.__current_scope)

    def exitValueExp(self, ctx: OpenSCENARIO2Parser.ValueExpContext):
        self._pop_parent(ctx)

    def enterListConstructor(self, ctx: OpenSCENARIO2Parser.ListConstructorContext):
        parent = self._push_parent(ctx)
        node = ast_node.ListExpression()
        self._open_node(ctx, node, self.__current_scope)

    def exitListConstructor(self, ctx: OpenSCENARIO2Parser.ListConstructorContext):
        self._pop_parent(ctx)

    def enterRangeConstructor(self, ctx: OpenSCENARIO2Parser.RangeConstructorContext):
        parent = self._push_parent(ctx)
        node = ast_node.RangeExpression()
        self._open_node(ctx, node, self.__current_scope)

    def exitRangeConstructor(self, ctx: OpenSCENARIO2Parser.RangeConstructorContext):
        self._pop_parent(ctx)

    def enterIdentifierReference(self, ctx: OpenSCENARIO2Parser.IdentifierReferenceContext):
        parent = self._push_parent(ctx)

        field_name = []
        scope = None
        for fn in ctx.fieldName():
            name = fn.Identifier().getText()
            if scope is None:
                scope = self.__current_scope.resolve(name) or None
            else:
                if issubclass(type(scope), TypedSymbol):
                    if self.__current_scope.resolve(scope.type):
                        scope = self.__current_scope.resolve(scope.type)
                    if name in getattr(scope, "symbols", {}):
                        if scope.symbols[name].value:
                            scope = scope.symbols[name].value
                else:
                    scope = self.__current_scope.resolve(scope)

            field_name.append(name)

        id_name = ".".join(field_name)

        node = ast_node.IdentifierReference(id_name)
        self._open_node(ctx, node, scope)

    def exitIdentifierReference(self, ctx: OpenSCENARIO2Parser.IdentifierReferenceContext):
        self._pop_parent(ctx)

    def enterArgumentListSpecification(self, ctx: OpenSCENARIO2Parser.ArgumentListSpecificationContext):
        pass

    def exitArgumentListSpecification(self, ctx: OpenSCENARIO2Parser.ArgumentListSpecificationContext):
        pass

    # -------- Arguments --------
    def _parse_type_text(self, t: str):
        t = (t or "").trim() if hasattr((t or ""), "trim") else (t or "").strip()
        if t.startswith("list<") and t.endswith(">"):
            inner = t[5:-1].strip()
            return "list", inner
        return "plain", t

    def enterArgumentSpecification(self, ctx: OpenSCENARIO2Parser.ArgumentSpecificationContext):
        self.__ensure_scope(ctx)
        parent = self._push_parent(ctx)

        argument_name = ctx.argumentName().getText()
        raw_type      = ctx.typeDeclarator().getText()
        default_value = ctx.defaultValue().getText() if ctx.defaultValue() else None

        kind, payload = self._parse_type_text(raw_type)
        if kind == "plain":
            if payload not in ("int","uint","float","bool","string"):
                resolved = self.__current_scope.resolve(payload) or self.__resolve_global_symbol(payload)
                if not resolved:
                    LOG_WARNING(f"Argument Type {raw_type} not resolved yet (deferring).", ctx.start)
        else:
            elem = payload
            if elem not in ("int","uint","float","bool","string"):
                resolved = self.__current_scope.resolve(elem) or self.__resolve_global_symbol(elem)
                if not resolved:
                    LOG_WARNING(f"Argument Element Type {elem} not resolved yet (deferring).", ctx.start)

        existing = getattr(self.__current_scope, "symbols", {}).get(argument_name)
        if isinstance(existing, ArgumentSpecificationSymbol):
            if getattr(existing, "type", None) and existing.type != raw_type:
                LOG_WARNING(
                    f"Argument '{argument_name}' re-declared with different type "
                    f"({raw_type}); keeping first: {existing.type}.", ctx.start
                )
            if default_value is not None and getattr(existing, "default", None) not in (None, default_value):
                LOG_WARNING(
                    f"Argument '{argument_name}' re-declared with different default; keeping first.", ctx.start
                )
            self.__current_scope = existing
        else:
            arg = ArgumentSpecificationSymbol(argument_name, self.__current_scope, raw_type, default_value)
            arg = self.__current_scope.define(arg, ctx.start) or arg
            self.__current_scope = arg

        node = ast_node.Argument(argument_name, raw_type, default_value)
        self._open_node(ctx, node, self.__current_scope)

    def exitArgumentSpecification(self, ctx: OpenSCENARIO2Parser.ArgumentSpecificationContext):
        self._pop_parent(ctx)
        self._leave_scope(ctx)

    def enterArgumentName(self, ctx: OpenSCENARIO2Parser.ArgumentNameContext):
        pass

    def exitArgumentName(self, ctx: OpenSCENARIO2Parser.ArgumentNameContext):
        pass

    def enterArgumentList(self, ctx: OpenSCENARIO2Parser.ArgumentListContext):
        pass

    def exitArgumentList(self, ctx: OpenSCENARIO2Parser.ArgumentListContext):
        pass

    def enterPositionalArgument(self, ctx: OpenSCENARIO2Parser.PositionalArgumentContext):
        parent = self._push_parent(ctx)
        node = ast_node.PositionalArgument()
        self._open_node(ctx, node, self.__current_scope)

    def exitPositionalArgument(self, ctx: OpenSCENARIO2Parser.PositionalArgumentContext):
        self._pop_parent(ctx)

    def enterNamedArgument(self, ctx: OpenSCENARIO2Parser.NamedArgumentContext):
        parent = self._push_parent(ctx)
        argument_name = ctx.argumentName().getText()
        node = ast_node.NamedArgument(argument_name)
        self._open_node(ctx, node, self.__current_scope)

    def exitNamedArgument(self, ctx: OpenSCENARIO2Parser.NamedArgumentContext):
        self._pop_parent(ctx)

    # -------- Physical literals / numbers --------
    def enterPhysicalLiteral(self, ctx: OpenSCENARIO2Parser.PhysicalLiteralContext):
        parent = self._push_parent(ctx)
        unit_name = ctx.Identifier().getText()
        value = None
        if ctx.FloatLiteral():
            value = ctx.FloatLiteral().getText()
        else:
            value = ctx.integerLiteral().getText()

        scope = self.__current_scope.resolve(unit_name)
        if not (scope and isinstance(scope, UnitSymbol)):
            LOG_WARNING("Unit " + unit_name + " is not defined!", ctx.start)

        node = ast_node.PhysicalLiteral(unit_name, value)
        self._open_node(ctx, node, self.__current_scope)

        if ctx.FloatLiteral():
            parent2 = self._push_parent(ctx)
            float_value = ctx.FloatLiteral().getText()
            value_node = ast_node.FloatLiteral(float_value)
            self._open_node(ctx, value_node, self.__current_scope)
            self._pop_parent(ctx)

    def exitPhysicalLiteral(self, ctx: OpenSCENARIO2Parser.PhysicalLiteralContext):
        self._pop_parent(ctx)

    def enterIntegerLiteral(self, ctx: OpenSCENARIO2Parser.IntegerLiteralContext):
        parent = self._push_parent(ctx)
        value = None
        type = "uint"
        if ctx.UintLiteral():
            value = ctx.UintLiteral().getText()
            type = "uint"
        elif ctx.HexUintLiteral():
            value = ctx.HexUintLiteral().getText()
            type = "hex"
        elif ctx.IntLiteral():
            value = ctx.IntLiteral().getText()
            type = "int"

        node = ast_node.IntegerLiteral(type, value)
        self._open_node(ctx, node, self.__current_scope)

    def exitIntegerLiteral(self, ctx: OpenSCENARIO2Parser.IntegerLiteralContext):
        self._pop_parent(ctx)


del OpenSCENARIO2Parser
