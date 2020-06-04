from typing import List, Tuple
import numpy as np
import torch
import itertools
import gensim
import json
import torch.nn.functional as F
import copy
import nltk
from nltk.stem.snowball import SnowballStemmer


class CodeNamesAgent(object):
    def __init__(self,
                 cuda_device: int = 0,
                 gensim_embeddings: str = 'embeddings/glove.42B.300d.w2vformat.txt',
                 maximal_combination_size: int = 999) -> None:
        self.MAXIMAL_SIZE_OF_COMBINATIONS = maximal_combination_size
        if "glove" in gensim_embeddings:
            self.embeddings = gensim.models.KeyedVectors.load_word2vec_format(gensim_embeddings, limit=200000)
        else:
            self.embeddings = gensim.models.KeyedVectors.load_word2vec_format(gensim_embeddings, binary=True)
        self.device = torch.device(cuda_device)
        self.vocabulary = self.embeddings.index2word
        self.stemmer = SnowballStemmer("english")
        self.stem_vocabulary = [self.stemmer.stem(word) for word in self.vocabulary]

    def get_clue(self, good_words: List[str],
                 bad_words: List[str] = [],
                 neutral_words: List[str] = [],
                 mine: str = None) -> Tuple[str, int]:
        if type(mine) is str:
            mine = [mine]
        else:
            mine = []
        aimed_words, the_word = self.flow(good_words, bad_words, mine, neutral_words)

        print(the_word, aimed_words)

        return (the_word, len(aimed_words))

    # check_capacity()
    def reduceVocabulary(self, board_words):
        # Keep only legitimate words in vocabulary

        stemWords = [self.stemmer.stem(word) for word in board_words]

        possible_vocabulary = copy.deepcopy(self.vocabulary)

        #         stem_vocabulary = [self.stemmer.stem(word) for word in new_vocabulary]
        words_to_remove = []
        for s_w_v, w_v in zip(self.stem_vocabulary, possible_vocabulary):
            for b in stemWords:
                if (s_w_v == b):
                    if w_v in possible_vocabulary:
                        words_to_remove.append(w_v)

        possible_vocabulary = list(filter(lambda x: x not in words_to_remove, possible_vocabulary))

        possible_vocabulary = list(
            filter(lambda x: "#" not in x and "_" not in x and "_" not in x, possible_vocabulary))
        possible_vocabulary = list(filter(lambda x: "/" not in x, possible_vocabulary))
        possible_vocabulary = list(filter(lambda x: "@" not in x, possible_vocabulary))
        possible_vocabulary = list(filter(lambda x: "." not in x, possible_vocabulary))

        #     new_vocabulary = list(filter(lambda x: x.lower() not in board_words, new_vocabulary))
        #     new_vocabulary = list(filter(lambda x: not x.isupper(), new_vocabulary))
        #     new_vocabulary = list(filter(lambda x: not any(self.isupper() for c in x), new_vocabulary))

        # Keep the maximal amount words in the vocabulary
        possible_vocabulary = possible_vocabulary[:20000]
        return possible_vocabulary

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

    def wordsAgentAimsFor(self, combinations, best_combination, good_words):
        aim_for_words = []
        for i, w in enumerate(combinations[best_combination].tolist()):
            if w > 0:
                aim_for_words.append(i)
        return aim_for_words, list(map(lambda x: good_words[x], aim_for_words))

    def flow(self, good_words, bad_words, mine, neutral_words):
        board_words = good_words + bad_words + mine + neutral_words
        vocabulary = self.reduceVocabulary(board_words)

        # set up my words vectors as tensors
        my_words = torch.tensor(self.embeddings[good_words]).to(self.device)

        # all other words from vocabulary
        full_vocabulary = torch.tensor(self.embeddings[vocabulary]).to(self.device)

        # words which we need to be very far from
        be_far_from_words = mine + bad_words
        away_from_these_vectors = torch.tensor(self.embeddings[be_far_from_words]).to(self.device)

        # prepare the vectors for cosine similarity
        my_words = F.normalize(my_words, p=2, dim=1)
        full_vocabulary = F.normalize(full_vocabulary, p=2, dim=1)
        away_from_these_vectors = F.normalize(away_from_these_vectors, p=2, dim=1)

        good_words2board_words = torch.matmul(my_words, full_vocabulary.permute(1, 0))

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

        similarity_of_bed_words = torch.matmul(away_from_these_vectors, full_vocabulary.permute(1, 0))
        # add extra points to the bed words
        similarity_of_bed_words[0, :] += 0.1

        # Go over all combinations and find the best combination which isn't close to the bed close
        found_word = False
        for ii, index_of_combination in enumerate(best_combinations.tolist()):
            if index_of_combination == 0: continue
            # check the maximal word to
            word_index = combination_best_index[index_of_combination].item()
            the_word = vocabulary[word_index]
            #             print(the_word)

            # get word vector
            # word_vector = self.embeddings[the_word]

            # the words which the agent is trying to convey to its team members
            index_aimed_words, aimed_words = self.wordsAgentAimsFor(combinations, index_of_combination, good_words)
            #             print(aimed_words)

            # get the minimal similarity to our desired words
            similarity_desired_words = good_words2board_words[
                index_aimed_words, combination_best_index[index_of_combination].item()]

            minimal_similarity = similarity_desired_words.min().item()
            maximalSimilarity2BedWords = similarity_of_bed_words[:, index_of_combination].max().item()

            if minimal_similarity > maximalSimilarity2BedWords:
                found_word = True
                break

        if found_word:
            print(aimed_words, the_word)
            print(similarity_desired_words)
            print(bad_words)
            print(similarity_of_bed_words[:, index_of_combination])
            return (aimed_words, the_word)
        else:
            return ([], "dummy")


if __name__ == "__main__":
    c = CodeNamesAgent(gensim_embeddings="embeddings/glove.42B.300d.w2vformat.txt")

    good_words = [
        "dad",

        "library",
        "way",
        "selection",
        "industry",
        "length"]

    neutral_words = ["student",
                     "movie",
                     "bath",
                     "blood",
                     "poet",
                     "setting",
                     "description",
                     "pollution",
                     "initiative"]

    mine = "bedroom"

    bad_words = ["photo",
                 "combination",
                 "housing",
                 "media",
                 "vehicle",
                 "communication",
                 "inspector"]

    # board_words = good_words + bad_words + neutral_words + mine

    board_words = {}
    board_words['good_words'] = None
    board_words['neutral_words'] = None
    board_words['mine_word'] = None
    board_words['bad_words'] = None
    c.get_clue(good_words=good_words, neutral_words=neutral_words, bad_words=bad_words, mine=mine)
