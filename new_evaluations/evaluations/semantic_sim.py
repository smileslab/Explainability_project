import math
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import numpy as np
import os, sys
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, modified_precision
tf.disable_v2_behavior()
import gensim.downloader as api
from gensim.models import Doc2Vec, KeyedVectors
from tqdm import tqdm
import scipy
import string
import re
from nltk.corpus import stopwords
from nltk.translate.meteor_score import single_meteor_score
import json

# get cosine similairty matrix
def cos_sim(input_vectors):
    similarity = cosine_similarity(input_vectors)
    return similarity

# get topN similar sentences
def get_top_similar(sentence, sentence_list, similarity_matrix, topN):
    # find the index of sentence in list
    index = sentence_list.index(sentence)
    # get the corresponding row in similarity matrix
    similarity_row = np.array(similarity_matrix[index, :])
    # get the indices of top similar
    indices = similarity_row.argsort()[-topN:][::-1]
    indices = indices[1:]
#    for i in indices:
#        print("{} : {} : {}".format(i, similarity_row[i], sentence_list[i]))
    return pd.DataFrame([[i, similarity_row[i], sentence_list[i]] for i in indices], columns=["indx", "sim", "text"])

def average_precision(ref, pred):
	if not ref:
		return 0.0

	score = 0.0
	num_hits = 0.0
	for i,p in enumerate(pred):
		if p in ref and p not in pred[:i]:
			num_hits += 1.0
			score += num_hits / (i + 1.0)

	return score / max(1.0, len(ref))

def MAP(ref, pred):
	avg_precision = average_precision(ref=ref, pred=pred)
	return avg_precision


def MRR(ref, pred):
	score = 0.0
	for rank, item in enumerate(pred):
		if item in ref:
			score = 1.0 / (rank + 1.0)
	return score

def cosine_distance_wordembedding_method(s1, s2):
    import scipy
    unknown = []
    vector_1, vector_2 = [], []
    for word in preprocess(s1):
        if word in word_vectors:
            vector_1.append(word_vectors[word])
        else:
            unknown.append(word)

    for word in preprocess(s2):
        if word in word_vectors:
            vector_2.append(word_vectors[word])
        else:
            unknown.append(word)
    if vector_2==[] or vector_1==[]:
        #print("\n{}\n{}\n".format(s1, s2))
        return 0
    cosine = scipy.spatial.distance.cosine(np.mean(vector_1, axis=0), np.mean(vector_2, axis=0))
#    print("Sentences compared:\n{}\n{}".format(s1, s2))
#    print("Unknown words: {}".format(unknown))
#    print('Word Embedding method with a cosine distance asses that our two sentences are similar to',round((1-cosine)*100,2),'%\n')
    return round((1-cosine), 4)


def preprocess(raw_text):
    # keep only words
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

    # convert to lower case and split
    words = letters_only_text.lower().split()

    # remove stopwords
    stopword_set = set(stopwords.words("english"))
    cleaned_words = list(set([w for w in words if w not in stopword_set]))
    return cleaned_words


def load_json(json_dir):
    with open(json_dir, 'r') as f:
        data = json.load(f)
        return data


if __name__=="__main__":
        word_vectors = KeyedVectors.load_word2vec_format('/projects/smiles/Bradley/google_w2v/GoogleNews-vectors-negative300.bin.gz', binary=True)
        base_data = []

######   CHANGE FILE NAMES TO RELEVANT TO YOUR RUN        #########
        for file_name in [
"/scratch/users/bfrink/new_dtuner/dtuner/output_frink_answer5_-1_DataTuner_No_FC_No_FS/2021-12-29_16-29-13/generated.json",
"/scratch/users/bfrink/new_dtuner/dtuner/output_frink_answer5_0_DataTuner_No_FC_No_FS/2021-12-29_16-29-22/generated.json",
"/scratch/users/bfrink/new_dtuner/dtuner/output_frink_answer5_1_DataTuner_No_FC_No_FS/2021-12-29_16-29-34/generated.json",
"/scratch/users/bfrink/new_dtuner/dtuner/output_frink_answer5_2_DataTuner_No_FC_No_FS/2021-12-29_16-29-45/generated.json",
"/scratch/users/bfrink/new_dtuner/dtuner/output_frink_answer5_3_DataTuner_No_FC_No_FS/2021-12-29_16-30-01/generated.json",
"/scratch/users/bfrink/new_dtuner/dtuner/output_frink_answer5_4_DataTuner_No_FC_No_FS/2021-12-29_16-30-14/generated.json"
]:
                print("============= {} =============".format(file_name))
                try:
                    data = load_json(file_name)
                except:
                    print("PROBLEM")
                    continue
                #best_sentences = ["" for i in range(len(data))]
                sentence_scores = []
                #scores_cos = [[] for i in range(len(data))]
                scores_bleu = [[] for i in range(len(data))]
                scores_bleu_1 = [[] for i in range(len(data))]
                scores_bleu_2 = [[] for i in range(len(data))]
                scores_bleu_3 = [[] for i in range(len(data))]
                scores_meteor = [[] for i in range(len(data))]
                #scores_map = [[] for i in range(len(data))]
                #scores_mrr = [[] for i in range(len(data))]
                cnt = 0
                for i in tqdm(data):
                        for k in i['ref         ']:
                 #               scores_cos[cnt].append(cosine_distance_wordembedding_method(i["original_ref"], k))
                                scores_meteor[cnt].append(single_meteor_score(i["original_ref"], k))
                                scores_bleu[cnt].append(sentence_bleu([i["original_ref"].translate(str.maketrans('', '', string.punctuation))], k.translate(str.maketrans('', '', string.punctuation))))
                                scores_bleu_1[cnt].append(float(modified_precision([i["original_ref"].translate(str.maketrans('', '', string.punctuation))], k.translate(str.maketrans('', '', string.punctuation)), n=1)))
                                scores_bleu_2[cnt].append(float(modified_precision([i["original_ref"].translate(str.maketrans('', '', string.punctuation))], k.translate(str.maketrans('', '', string.punctuation)), n=2)))
                                scores_bleu_3[cnt].append(float(modified_precision([i["original_ref"].translate(str.maketrans('', '', string.punctuation))], k.translate(str.maketrans('', '', string.punctuation)), n=3)))
                 #               scores_map[cnt].append(MAP(set(i['original_ref']), k))
                 #               scores_mrr[cnt].append(MRR(set(i['original_ref']), k))
                        #print(scores_cos)
                 #       best_indx = scores_bleu[cnt].index(max(scores_bleu[cnt]))
                 #       best_sentences[cnt] = i['ref         '][best_indx]
                 #       scores_cos[cnt] = np.mean(scores_cos[cnt])
                        scores_meteor[cnt] = np.mean(scores_meteor[cnt])
                        scores_bleu[cnt] = np.mean(scores_bleu[cnt])
                        scores_bleu_1[cnt] = np.mean(scores_bleu_1[cnt])
                        scores_bleu_2[cnt] = np.mean(scores_bleu_2[cnt])
                        scores_bleu_3[cnt] = np.mean(scores_bleu_3[cnt])

                 #       scores_map[cnt] = np.mean(scores_map[cnt])
                 #       scores_mrr[cnt] = np.mean(scores_mrr[cnt])
                 #       sentence_scores.append([i["original_ref"], scores_cos[cnt], scores_meteor[cnt], scores_bleu[cnt], scores_map[cnt], scores_mrr[cnt]])
                        sentence_scores.append([i["original_ref"], scores_meteor[cnt], scores_bleu[cnt], scores_bleu_1[cnt], scores_bleu_2[cnt], scores_bleu_3[cnt]])
                        cnt += 1
                out_sent_scores = pd.DataFrame(sentence_scores, columns=["sentences","meteor", "bleu-4", "bleu-1", "bleu-2", "bleu-3"])
                out_sent_scores.to_csv("sent_scores_{}.csv".format(file_name.split("/")[2]))

#                out_best_sents = pd.DataFrame(best_sentences, columns=["gen_sentence"])
#                out_best_sents.to_csv("gen_sentences_{}_DEMO.csv".format(file_name.split("/")[2]))

               # print("Average Cosine Similarity for all generated from {}: {}".format(file_name.split("/")[1], round(np.mean(scores_cos), 4)))
                print("Average BLEU-4 for all generated from {}: {}".format(file_name.split("/")[1], round(np.mean(scores_bleu), 4)))
                print("Average METEOR for all generated from {}: {}".format(file_name.split("/")[1], round(np.mean(scores_meteor), 4)))
                print("Average BLEU-1 for all generated from {}: {}".format(file_name.split("/")[1], round(np.mean(scores_bleu_1), 4)))
                print("Average BLEU-2 for all generated from {}: {}".format(file_name.split("/")[1], round(np.mean(scores_bleu_2), 4)))
                print("Average BLEU-3 for all generated from {}: {}".format(file_name.split("/")[1], round(np.mean(scores_bleu_3), 4)))

               # print("Average MAP for all generated from {}: {}".format(file_name.split("/")[1], round(np.mean(scores_map), 4)))
               # print("Average MRR for all generated from {}: {}".format(file_name.split("/")[1], round(np.mean(scores_mrr), 4)))
                print("\n\n")

