import numpy as np
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE, Laplace
import pandas as pd
import math


def perplexity_calc(ref, gen, n=1):
    """
    Perplexity calculation between original reference sentences and generated sentences.
    ref: List, original reference sentences
    gen: List, generated sentences
    n: int, determans the order of n-grams.
    """
    out_scores = []
    train_sentences = ref
    tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent)))
                    for sent in train_sentences]

    train_data, padded_vocab = padded_everygram_pipeline(n, tokenized_text)
    model = Laplace(n)
    model.fit(train_data, padded_vocab)

    test_sentences = gen
    tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent)))
                    for sent in test_sentences]

    test_data, _ = padded_everygram_pipeline(n, tokenized_text)

    for i, test in enumerate(test_data):
        out_scores.append(model.perplexity(test))

    return sum(out_scores)/len(out_scores)


def read_json(_dir):
    fp = open(_dir, 'r')
    data = pd.read_json(fp)
    generated = data["ref         "]
    original = data["original_ref"]
    return original, generated


def calculate_scores(_dir, score_name):
    original, generated = read_json(_dir)
    score = []
    for indx, orig in enumerate(original):
        if score_name == "perplexity":
            score.append(perplexity_calc(orig, generated[indx]))
    return score


if __name__ == "__main__":
    num_enrich = 5
    version = "V1.3"
    _base = "/Users/BradleyFrink/Desktop/Q-A generation outputs/{}/Questions/".format(version)
    _dirs = ["unenriched_questions_generated_{}.json".format(version)]
    fp = open("score_{}.txt".format(version), "w+")
    for i in range(num_enrich):
        _dirs.append("enrich_{}_questions_generated_{}.json".format(i, version))
    for _dir in _dirs:
        score = calculate_scores(_base+_dir, "perplexity")
        temp = "{} : {}\n".format(_dir, sum(score)/len(score))
        fp.write(temp)
