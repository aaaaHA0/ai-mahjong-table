class MahjongEngineError(RuntimeError):
    """Base class for deterministic engine failures."""


class IllegalActionError(MahjongEngineError):
    pass


class StateInvariantError(MahjongEngineError):
    pass


class RuleConfigurationError(MahjongEngineError):
    pass


class RuleExecutionError(MahjongEngineError):
    pass


class AgentExecutionError(MahjongEngineError):
    pass


class ReplayCompatibilityError(MahjongEngineError):
    pass
