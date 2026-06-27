from enum import StrEnum


class GamePhase(StrEnum):
    INITIAL = "initial"
    WAITING_FOR_DRAW = "waiting_for_draw"
    WAITING_FOR_DISCARD = "waiting_for_discard"
    WAITING_FOR_DISCARD_RESPONSES = "waiting_for_discard_responses"
    WAITING_FOR_ROB_KONG_RESPONSES = "waiting_for_rob_kong_responses"
    WAITING_FOR_KONG_REPLACEMENT = "waiting_for_kong_replacement"
    TERMINAL = "terminal"
