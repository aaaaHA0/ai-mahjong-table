from __future__ import annotations

from enum import IntEnum


class Seat(IntEnum):
    EAST = 0
    NORTH = 1
    WEST = 2
    SOUTH = 3

    def next(self) -> "Seat":
        return Seat((int(self) + 1) % 4)


ALL_SEATS = tuple(Seat)

