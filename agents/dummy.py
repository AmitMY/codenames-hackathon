from typing import List, Tuple
from .base import CodenamesAgent


class DummyCodenamesAgent(CodenamesAgent):
    def get_clue(self, good_words: List[str],
                 bad_words: List[str] = [],
                 neutral_words: List[str] = [],
                 mine: str = None) -> Tuple[str, int]:
        return "dummy", 3
