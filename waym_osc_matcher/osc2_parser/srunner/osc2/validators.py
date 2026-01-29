from .osc2_parser.OpenSCENARIO2Listener import OpenSCENARIO2Listener

class ActorsCollect(OpenSCENARIO2Listener):
    """
    Pass 1: collect declared actors.
    """
    def __init__(self, case_insensitive: bool = True):
        self.case_insensitive = case_insensitive
        self.actors = set()

    def _norm(self, s: str) -> str:
        return s.casefold() if self.case_insensitive else s

    # actorDeclaration : 'actor' actorName ...
    def exitActorDeclaration(self, ctx):
        name = ctx.actorName().getText()
        self.actors.add(self._norm(name))


class ActorsValidate(OpenSCENARIO2Listener):
    """
    Pass 2: validate that action headers reference existing actors.
    Works with your current grammar's dotted header:
      action <qualifiedBehaviorName> ...
      where qualifiedBehaviorName := (actorName '.')? behaviorName
    """
    def __init__(self, actor_names, error_reporter, case_insensitive: bool = True):
        self.actor_names = actor_names
        self.err = error_reporter  # must have .report(line, col, msg)
        self.case_insensitive = case_insensitive

    def _norm(self, s: str) -> str:
        return s.casefold() if self.case_insensitive else s

    # actionDeclaration : 'action' qualifiedBehaviorName ...
    def exitActionDeclaration(self, ctx):
        qn = ctx.qualifiedBehaviorName()
        # qualifiedBehaviorName : (actorName '.')? behaviorName
        actor_ctx = qn.actorName()  # may be None
        if actor_ctx is None:
            return
        raw = actor_ctx.getText()
        if self._norm(raw) not in self.actor_names:
            t = ctx.start
            line = getattr(t, "line", None)
            col  = getattr(t, "column", None)
            self.err.report(line, col, f"actorName: {raw} is not defined!")