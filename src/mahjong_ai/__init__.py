"""Mahjong table engine and AI training primitives."""

from mahjong_ai.game.match import MatchController
from mahjong_ai.game.table import TableEngine
from mahjong_ai.rules.loader import load_rule_plugin

__all__ = ["MatchController", "TableEngine", "load_rule_plugin"]

