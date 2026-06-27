from __future__ import annotations

from dataclasses import dataclass


SUIT_PREFIXES = ("W", "B", "T")
HONOR_PREFIXES = ("F", "J")


@dataclass(frozen=True, order=True)
class TileType:
    code: str

    def __post_init__(self) -> None:
        if not is_valid_tile_code(self.code):
            raise ValueError(f"invalid Mahjong tile code: {self.code!r}")

    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True, order=True)
class PhysicalTile:
    tile_type: TileType
    copy_id: int

    def __post_init__(self) -> None:
        if self.copy_id not in range(4):
            raise ValueError("copy_id must be in range 0..3")

    @property
    def id(self) -> str:
        return f"{self.tile_type.code}#{self.copy_id}"


def is_valid_tile_code(code: str) -> bool:
    if len(code) != 2 or not code[1].isdigit():
        return False
    value = int(code[1])
    if code[0] in SUIT_PREFIXES:
        return 1 <= value <= 9
    if code[0] == "F":
        return 1 <= value <= 4
    if code[0] == "J":
        return 1 <= value <= 3
    return False


def all_tile_types() -> tuple[TileType, ...]:
    codes = [
        *(f"{prefix}{value}" for prefix in SUIT_PREFIXES for value in range(1, 10)),
        *(f"F{value}" for value in range(1, 5)),
        *(f"J{value}" for value in range(1, 4)),
    ]
    return tuple(TileType(code) for code in codes)


def standard_136_tiles() -> list[PhysicalTile]:
    return [
        PhysicalTile(tile_type, copy_id)
        for tile_type in all_tile_types()
        for copy_id in range(4)
    ]

