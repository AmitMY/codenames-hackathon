import numpy as np
import torch
import itertools
import gensim

MAXIMAL_SIZE_OF_COMBINATIONS = 99
w2v = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# corpus = api.load('text8')  # download the corpus and return it opened as an iterable
# model = Word2Vec(corpus)  # train a model from the corpus
# w2v = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# "http://nlp.stanford.edu/data/glove.6B.zip"
# "./glove.6B.zip"

# glove = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt")

device = torch.device(0)

GOOD_WORDS = ["entry",
              "context",
              "world",
              "flight",
              "payment",
              "medicine",
              "strategy",
              "chest"]

GOOD_WORDS = ["world",
              "flight",
              "chest"]

netural_words = ["student",
                 "movie",
                 "bath",
                 "blood",
                 "poet",
                 "setting",
                 "description",
                 "pollution",
                 "initiative"]

worst_word = ["bedroom"]

BAD_WORDS = ["photo",
             "combination",
             "housing",
             "media",
             "vehicle",
             "communication",
             "inspector"]

board_words = GOOD_WORDS + BAD_WORDS + netural_words + worst_word

EMBEDDINGS = w2v
vocabulary_size = len(EMBEDDINGS.index2entity)
vocabulary = EMBEDDINGS.index2entity


# check_capacity()

def reduceVocabulary():
    global vocabulary
    # Keep only legitimate words in vocabulary
    vocabulary = list(filter(lambda x: "#" not in x and "_" not in x and "_" not in x, vocabulary))
    vocabulary = list(filter(lambda x: "/" not in x, vocabulary))
    vocabulary = list(filter(lambda x: "@" not in x, vocabulary))
    vocabulary = list(filter(lambda x: "." not in x, vocabulary))
    vocabulary = list(filter(lambda x: x.lower() not in board_words, vocabulary))
    vocabulary = list(filter(lambda x: not x.isupper(), vocabulary))
    vocabulary = list(filter(lambda x: not any(c.isupper() for c in x), vocabulary))
    # Keep the maximal amount words in the vocabulary
    vocabulary = vocabulary[:20000]


def setBinaryMatrix():
    # it's a binary matrix of size (2^N X N) where N is the my number of words
    combinations = list(map(list, itertools.product([0, 1], repeat=len(GOOD_WORDS))))
    # flip combinations to ve aligned with the GOOD words
    combinations = list(map(lambda x: x[::-1], combinations))
    combinations = list(filter(lambda x: sum(x) < MAXIMAL_SIZE_OF_COMBINATIONS, combinations))
    combinations = torch.tensor(combinations).to(device=device).float()
    return combinations


def getBestWordForCombination(index, combinations, combination_best_index):
    comb = combinations[index].tolist()
    words = []
    for i, w in zip(comb, GOOD_WORDS):
        if i > 0:
            words.append(w)
    print(words)
    word_index = combination_best_index[index].item()
    the_word = vocabulary[word_index]
    return the_word


def wordsAgentAimsFor(combinations, best_combination):
    aim_for_words = []
    for i, w in enumerate(combinations[best_combination].tolist()):
        if w > 0:
            aim_for_words.append(i)
    return aim_for_words, list(map(lambda x: GOOD_WORDS[x], aim_for_words))


def main():
    reduceVocabulary()

    # set up my words vectors as tensors
    my_words = torch.tensor(EMBEDDINGS[GOOD_WORDS]).to(device)
    # all other words from vocabulary
    full_vocabulary = torch.tensor(EMBEDDINGS[vocabulary]).to(device)
    # similarities of my words to all other words in the vocabulary
    good_words2board_words = torch.matmul(my_words, full_vocabulary.permute(1, 0))

    # the mean value of similarity is 0.596. I decide to remove 2 so the mean is negative
    good_words2board_words -= 1

    # set up a matrix of all possible combinations
    combinations = setBinaryMatrix()

    # get the sum of similarities of all combinations
    combinations_good_words = torch.matmul(combinations, good_words2board_words)

    # maximal value per word in the vocabulary
    combination_values, combination_best_index = combinations_good_words.max(1)

    # get the maximal combination
    max_in_rows, best_combinations = torch.sort(combination_values, dim=-1, descending=True)

    assert combinations_good_words[best_combinations[0].item(), combination_best_index[
        best_combinations[0].item()].item()].item() == combinations_good_words.max().item()

    # words which we need to be very far from
    be_far_from_words = worst_word + BAD_WORDS
    away_from_the_vectors = torch.tensor(EMBEDDINGS[be_far_from_words]).to(device)
    similarity_of_bed_words = torch.matmul(away_from_the_vectors, full_vocabulary.permute(1, 0))
    similarity_of_bed_words -= 1

    found_word = False
    for ii, index_of_combination in enumerate(best_combinations.tolist()):
        if index_of_combination == 0: continue
        # check the maximal word to
        word_index = combination_best_index[index_of_combination].item()
        the_word = vocabulary[word_index]
        print(the_word)

        # get word vector
        word_vector = EMBEDDINGS[the_word]

        # the words which the agent is trying to convey to its team members
        index_aimed_words, aimed_words = wordsAgentAimsFor(combinations, index_of_combination)

        # get the minimal similarity to our desired words
        similarity_desired_words = good_words2board_words[
            index_aimed_words, combination_best_index[index_of_combination].item()]

        minimal_similarity = similarity_desired_words.min().item()
        maximalSimilarity2BedWords = similarity_of_bed_words[:, index_of_combination].max().item()

        if minimal_similarity > maximalSimilarity2BedWords:
            found_word = True
            if any(the_word in x or x in the_word for x in aimed_words): continue
            break

    assert found_word
    print(aimed_words)
    print(the_word)


if __name__ == "__main__":
    main()
