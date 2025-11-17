"""Definition der möglichen Aktionen, die ein Agent in der Welt ausführen kann."""

from enum import Enum

class Action(Enum):
    MOVE_UP = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4
    GRAB = 5
    SHOOT_UP = 6
    SHOOT_DOWN = 7
    SHOOT_LEFT = 8
    SHOOT_RIGHT = 9

