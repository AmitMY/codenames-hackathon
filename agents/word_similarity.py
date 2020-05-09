from typing import List, Tuple
import numpy as np
import torch
import itertools
import gensim
import json
import torch.nn.functional as F


# "http://nlp.stanford.edu/data/glove.6B.zip"
# "./glove.6B.zip"
# glove = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt")

class CodeNamesAgent(object):
    def __init__(self,
                 cuda_device: int = 0,
                 gensim_embeddings: str = 'GoogleNews-vectors-negative300.bin',
                 maximal_combination_size: int = 999) -> None:
        self.MAXIMAL_SIZE_OF_COMBINATIONS = maximal_combination_size
        self.embeddings = gensim.models.KeyedVectors.load_word2vec_format(gensim_embeddings, binary=True)

        self.device = torch.device(cuda_device)
        self.vocabulary = self.embeddings.index2word

    def get_clue(self, good_words: List[str],
                 bad_words: List[str] = [],
                 neutral_words: List[str] = [],
                 mine: str = None) -> Tuple[str, int]:

        aimed_words, the_word = self.flow(good_words, bad_words, mine, neutral_words)

        print(the_word, len(aimed_words))

        return (the_word, len(aimed_words))

    # check_capacity()

    def reduceVocabulary(self, board_words):
        # Keep only legitimate words in vocabulary
        new_vocabulary = list(filter(lambda x: "#" not in x and "_" not in x and "_" not in x, self.vocabulary))
        new_vocabulary = list(filter(lambda x: "/" not in x, new_vocabulary))
        new_vocabulary = list(filter(lambda x: "@" not in x, new_vocabulary))
        new_vocabulary = list(filter(lambda x: "." not in x, new_vocabulary))
        new_vocabulary = list(filter(lambda x: x.lower() not in board_words, new_vocabulary))
        new_vocabulary = list(filter(lambda x: not x.isupper(), new_vocabulary))
        new_vocabulary = list(filter(lambda x: not any(c.isupper() for c in x), new_vocabulary))
        # todo add stemming of words

        v_appear_on_board = []
        for substring in board_words:
            v_appear_on_board.extend([string for string in new_vocabulary if substring in string])

        new_vocabulary = list(filter(lambda x: x not in v_appear_on_board, new_vocabulary))
        # Keep the maximal amount words in the vocabulary
        new_vocabulary = new_vocabulary[:20000]
        return new_vocabulary

    def setBinaryMatrix(self, size_of_good_words):
        # it's a binary matrix of size (2^N X N) where N is the my number of words
        combinations = list(map(list, itertools.product([0, 1], repeat=size_of_good_words)))
        # flip combinations to ve aligned with the GOOD words
        combinations = list(map(lambda x: x[::-1], combinations))
        # combinations = list(filter(lambda x: sum(x) < self.MAXIMAL_SIZE_OF_COMBINATIONS, combinations))
        combinations = torch.tensor(combinations).to(device=self.device).float()
        return combinations

    def getBestWordForCombination(self, index, combinations, combination_best_index, good_words):
        comb = combinations[index].tolist()
        words = []
        for i, w in zip(comb, good_words):
            if i > 0:
                words.append(w)
        print(words)
        word_index = combination_best_index[index].item()
        the_word = self.vocabulary[word_index]
        return the_word


    def wordsAgentAimsFor(self,combinations, best_combination,good_words):
        aim_for_words = []
        for i, w in enumerate(combinations[best_combination].tolist()):
            if w > 0:
                aim_for_words.append(i)
        return aim_for_words, list(map(lambda x: good_words[x], aim_for_words))


    def flow(self, good_words, bad_words, mine):
        vocabulary = self.reduceVocabulary(good_words)

        # set up my words vectors as tensors
        my_words = torch.tensor(self.embeddings[good_words]).to(self.device)
        # all other words from vocabulary
        full_vocabulary = torch.tensor(self.embeddings[vocabulary]).to(self.device)
        # similarities of my words to all other words in the vocabulary
        good_words2board_words = torch.matmul(my_words, full_vocabulary.permute(1, 0))

        # the mean value of similarity is 0.596. I decide to remove 2 so the mean is negative
        good_words2board_words -= 1
        good_words2board_words = good_words2board_words.clamp_max(4)

        # set up a matrix of all possible combinations
        combinations = self.setBinaryMatrix(len(good_words))

        # get the sum of similarities of all combinations
        combinations_good_words = torch.matmul(combinations, good_words2board_words)

        # maximal value per word in the vocabulary
        combination_values, combination_best_index = combinations_good_words.max(1)

        # get the maximal combination
        max_in_rows, best_combinations = torch.sort(combination_values, dim=-1, descending=True)

        assert combinations_good_words[best_combinations[0].item(), combination_best_index[
            best_combinations[0].item()].item()].item() == combinations_good_words.max().item()

        # words which we need to be very far from
        be_far_from_words = [mine] + bad_words
        away_from_the_vectors = torch.tensor(self.embeddings[be_far_from_words]).to(self.device)
        similarity_of_bed_words = torch.matmul(away_from_the_vectors, full_vocabulary.permute(1, 0))
        # similarity_of_bed_words -= 1

        found_word = False
        for ii, index_of_combination in enumerate(best_combinations.tolist()):
            if index_of_combination == 0: continue
            # check the maximal word to
            word_index = combination_best_index[index_of_combination].item()
            the_word = vocabulary[word_index]
            print(the_word)

            # for x in board_words:
            #     if (the_word in x) or (x in the_word):
            #         print("!!!",x)
            #         illeagal_word = True
            #         break
            # if illeagal_word:
            #     continue

            # get word vector
            word_vector = self.embeddings[the_word]

            # the words which the agent is trying to convey to its team members
            index_aimed_words, aimed_words = self.wordsAgentAimsFor(combinations, index_of_combination, good_words)

            # get the minimal similarity to our desired words
            similarity_desired_words = good_words2board_words[
                index_aimed_words, combination_best_index[index_of_combination].item()]

            minimal_similarity = similarity_desired_words.min().item()
            maximalSimilarity2BedWords = similarity_of_bed_words[:, index_of_combination].max().item()

            if minimal_similarity > maximalSimilarity2BedWords:
                found_word = True
                break

        assert found_word
        return aimed_words,the_word


if __name__ == "__main__":
    c = CodeNamesAgent(gensim_embeddings="../GoogleNews-vectors-negative300.bin")

    GOOD_WORDS = ["entry",
                  "context",
                  "world",
                  "flight",
                  "payment",
                  "medicine",
                  "strategy",
                  "chest"]

    neutral_words = ["student",
                     "movie",
                     "bath",
                     "blood",
                     "poet",
                     "setting",
                     "description",
                     "pollution",
                     "initiative"]

    mined = "bedroom"

    BAD_WORDS = ["photo",
                 "combination",
                 "housing",
                 "media",
                 "vehicle",
                 "communication",
                 "inspector"]

    # board_words = GOOD_WORDS + BAD_WORDS + neutral_words + mined

    board_words = {}
    board_words['good_words'] = None
    board_words['neutral_words'] = None
    board_words['mined_word'] = None
    board_words['bad_words'] = None
    c.get_clue(good_words=GOOD_WORDS, neutral_words=neutral_words, bad_words=BAD_WORDS, mine=mined)
