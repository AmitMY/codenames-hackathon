from typing import List, Tuple


class CodenamesAgent:
    def get_clue(self, good_words: List[str],
                 bad_words: List[str] = [],
                 neutral_words: List[str] = [],
                 mine: str = None) -> Tuple[str, int]:
        raise NotImplementedError("Must implement `get_clue` on CodenamesAgent")


