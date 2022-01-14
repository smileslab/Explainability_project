import json
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import string
import math
from tqdm import tqdm
from collections import OrderedDict


def detection_switch(arg, arg1):
    if arg == "Replayed":
        return replayed(arg1)
    elif arg == "Synthetic":
        return synthetic(arg1)
    elif arg == "Conversion":
        return conversion(arg1)
    elif arg == "Bonafide":
        return bonafide(arg1)
    elif arg == "Reverberation":
        return reverberation(arg1)
    elif arg == "Microphone":
        return microphone(arg1)
    else:
        arg1.append(arg)
        return asv(arg1)


def asv(arg):
    questions = []
    refs = []
    detected = arg.pop()
    questions.append("request(detected[{}])".format(detected))
    questions.append("request_explaination(detected[{}])".format(detected))
    refs.append("Was the audio sample of person {}?".format(detected))
    refs.append("How was the audio sample detected as being made by person {}?".format(detected))

    questions.append("request(detected[{}])".format(detected))
    questions.append("request_explaination(detected[{}])".format(detected))
    refs.append("Was the speaker in the audio sample of person {}?".format(detected))
    refs.append("How was person {} detected as the speaker of the audio?".format(detected))

    questions.append("request(detected[{}])".format(detected))
    questions.append("request_explaination(detected[{}])".format(detected))
    refs.append("Was person {} detected as the primary speaker of the audio sample?".format(detected))
    refs.append("How was person {} found to be the id of the speaker in the audio sample?".format(detected))

    questions.append("request(detected[{}])".format(detected))
    questions.append("request_explaination(detected[{}])".format(detected))
    refs.append("{} detected as the id of the primary speaker of the audio sample?".format(detected))
    refs.append("How was {} identified as the speaker?".format(detected))


    for i in arg:
        if "GTCC" not in i[0] and "MFCC" not in i[0]: continue
        questions.append("request(detected[{}], detected_by[{}], shap_value[{:.3f}])".format(detected, i[0], i[1]))
        questions.append("verify_attribute(detected[{}], detected_by[{}], shap_value[{:.3f}])".format(detected ,i[0], i[1]))
        questions.append("request_explaination(detected[{}], detected_by[{}], shap_value[{:.3f}])".format(detected, i[0], i[1]))
        refs.append("was the audio sample made by person {} detected by {}, with a shap value of {:.3f}?".format(detected, i[0], i[1]))
        refs.append("The audio sample made by person {} was detected by {}, with a shap value of {:.3f}, was the audio sample detected by other features?".format(detected, i[0], i[1]))
        refs.append("How was the audio sample made by person {} detected using {}, with a shap value of {:.3f}?".format(detected, i[0], i[1]))

        questions.append("request(detected[{}], detected_by[{}], shap_value[{:.3f}])".format(detected, i[0], i[1]))
        questions.append("verify_attribute(detected[{}], detected_by[{}], shap_value[{:.3f}])".format(detected ,i[0], i[1]))
        questions.append("request_explaination(detected[{}], detected_by[{}], shap_value[{:.3f}])".format(detected, i[0], i[1]))
        refs.append("was the speaker person {} detected by {}, with a shap value of {:.3f}?".format(detected, i[0], i[1]))
        refs.append("{} was found to be the speaker id, was the audio sample detected by {}, with a shap value of {:.3f}?".format(detected, i[0], i[1]))
        refs.append("How was {}, with a shap value of {:.3f}, used to detect person {} as the speaker of the audio sample?".format(i[0], i[1], detected))

        questions.append("request(detected[{}], detected_by[{}], shap_value[{:.3f}])".format(detected, i[0], i[1]))
        questions.append("verify_attribute(detected[{}], detected_by[{}], shap_value[{:.3f}])".format(detected ,i[0], i[1]))
        questions.append("request_explaination(detected[{}], detected_by[{}], shap_value[{:.3f}])".format(detected, i[0], i[1]))
        refs.append("was person {} detected by {}, with a shap value of {:.3f} detected as the speaker?".format(detected, i[0], i[1]))
        refs.append("person {} was detected as the speaker of the audio file, was the audio sample detected by {}?".format(detected, i[0], i[1]))
        refs.append("How did {} impact the detection of person {}, with a shap value of {:.3f}, as the primary speaker in the audio?".format(i[0], detected, i[1]))

        questions.append("request(detected[{}], detected_by[{}], shap_value[{:.3f}])".format(detected, i[0], i[1]))
        questions.append("verify_attribute(detected[{}], detected_by[{}], shap_value[{:.3f}])".format(detected ,i[0], i[1]))
        questions.append("request_explaination(detected[{}], detected_by[{}], shap_value[{:.3f}])".format(detected, i[0], i[1]))
        refs.append("was person {} detected as the speaker of the audio sample by {}, with a shap value of {:.3f}?".format(detected, i[0], i[1]))
        refs.append("{} was detected as the id of the speaker of this audio sample, was the id detected by {}, with a shap value of {:.3f}?".format(detected, i[0], i[1]))
        refs.append("How was {} detected as the id of the speaker using {}, with a shap value of {:.3f}?".format(detected, i[0], i[1]))

    return questions, refs


def conversion(arg):
    questions = []
    refs = []
    questions.append("request(detected[Spoofed], spoof_type[Conversion])")
    questions.append("verify_attribute(detected[Spoofed], spoof_type[Conversion])")
    questions.append("request_explaination(detected[Spoofed], spoof_type[Conversion])")

    refs.append("Was the spoofed audio sample a Conversion sample?")
    refs.append("The spoofed audio sample was Conversion, were there any other spoof types?")
    refs.append("How was the spoofed audio sample detected as a Conversion sample?")

    for i in arg:
        questions.append("request(detected[Conversion], detected_by[{}]".format(i))
        questions.append("verify_attribute(detected[Conversion], detected_by[{}])".format(i))

        refs.append("was the Conversion audio sample detected by {}?".format(i))
        refs.append("The Conversion audio sample was detected by {}, was the audio sample detected by other features?".format(i))

    return questions, refs


def synthetic(arg):
    questions = []
    refs = []
    questions.append("request(detected[Spoofed], spoof_type[TextToSpeech])")
    questions.append("verify_attribute(detected[Spoofed], spoof_type[TextToSpeech])")
    questions.append("request_explaination(detected[Spoofed], spoof_type[TextToSpeech])")

    refs.append("Was the spoofed audio sample a TextToSpeech sample?")
    refs.append("The spoofed audio sample was TextToSpeech, were there any other spoof types?")
    refs.append("How was the spoofed audio sample detected as a TextToSpeech sample?")

    for i in arg:
        questions.append("request_explaination(detected[TextToSpeech], detected_by[{}])".format(i))
        questions.append("verify_attribute(detected[TextToSpeech], detected_by[{}])".format(i))

        refs.append("was the TextToSpeech audio sample detected by {}?".format(i))
        refs.append("The TextToSpeech audio sample was detected by {}, was the audio sample detected by other features?".format(i))

    return questions, refs


def replayed(arg):
    questions = []
    refs = []
    questions.append("request(detected[Spoofed], spoof_type[Replayed])")
    questions.append("verify_attribute(detected[Spoofed], spoof_type[Replayed])")
    questions.append("request_explaination(detected[Spoofed], spoof_type[Replayed])")
    refs.append("Was the spoofed audio sample a Replayed sample?")
    refs.append("The spoofed audio sample was Replayed, were there any other spoof types?")
    refs.append("How was the spoofed audio sample detected as a Replayed sample?")

    questions.append("request(detected[Spoofed], spoof_type[Replayed])")
    questions.append("verify_attribute(detected[Spoofed], spoof_type[Replayed])")
    questions.append("request_explaination(detected[Spoofed], spoof_type[Replayed])")
    refs.append("Was the spoofed audio sample a Replayed sample?")
    refs.append("The spoofed audio sample was Replayed, were there any other spoof types?")
    refs.append("How was the spoofed audio classified as Replayed?")

    questions.append("request(detected[Spoofed], spoof_type[Replayed])")
    questions.append("verify_attribute(detected[Spoofed], spoof_type[Replayed])")
    questions.append("request_explaination(detected[Spoofed], spoof_type[Replayed])")
    refs.append("Was Replayed the spoofing type of the spoofed audio?")
    refs.append("The spoofed audio sample was Replayed, what other spoofing methods were present?")
    refs.append("How was Replayed determined to be the spoof type of the audio sample?")


    for i in arg:
        questions.append("request(detected[Replayed], detected_by[{}], shap_value[{:.3f}])".format(i[0], i[1]))
        questions.append("verify_attribute(detected[Replayed], detected_by[{}], shap_value[{:.3f}])".format(i[0], i[1]))
        questions.append("request_explaination(detected[Replayed], detected_by[{}], shap_value[{:.3f}])".format(i[0], i[1]))
        refs.append("was the Replayed audio sample detected by {}, with a shap value of {:.3f}?".format(i[0], i[1]))
        refs.append("The Replayed audio sample was detected by {}, with a shap value of {:.3f}, was the audio sample detected by other features?".format(i[0], i[1]))
        refs.append("How was {}, with a shap value of {:.3f}, determined to have an impact on classifying the Replayed sample?".format(i[0], i[1]))

        questions.append("request(detected[Replayed], detected_by[{}], shap_value[{:.3f}])".format(i[0], i[1]))
        questions.append("verify_attribute(detected[Replayed], detected_by[{}], shap_value[{:.3f}])".format(i[0], i[1]))
        questions.append("request_explaination(detected[Replayed], detected_by[{}], shap_value[{:.3f}])".format(i[0], i[1]))
        refs.append("was {}, with a shap value of {:.3f}, used to detect the audio sample as Replayed?".format(i[0], i[1]))
        refs.append("{}, with a shap value of {:.3f}, was used to determine that the audio sample was replayed, what other features were used?".format(i[0], i[1]))
        refs.append("How did {}, with a shap value of {:.3f}, help detect the Replayed sample?".format(i[0], i[1]))

        questions.append("request(detected[Replayed], detected_by[{}], shap_value[{:.3f}])".format(i[0], i[1]))
        questions.append("verify_attribute(detected[Replayed], detected_by[{}], shap_value[{:.3f}])".format(i[0], i[1]))
        questions.append("request_explaination(detected[Replayed], detected_by[{}], shap_value[{:.3f}])".format(i[0], i[1]))
        refs.append("was the replayed audio sample classified using {}, with a shap value of {:.3f}?".format(i[0], i[1]))
        refs.append("{}, with a shap value of {:.3f}, helped determine that the audio sample was replayed, what other features were used?".format(i[0], i[1]))
        refs.append("How was the replayed audio determined by {}, with a shap value of {:.3f}?".format(i[0], i[1]))


    return questions, refs


def bonafide(arg):
    questions = []
    refs = []
    questions.append("request(detected[Bonafide])")
    questions.append("request_explaination(detected[Bonafide])")
    refs.append("Was the audio sample a Bonafide sample?")
    refs.append("How was the audio sample detected as a Bonafide sample?")

    questions.append("request(detected[Bonafide])")
    questions.append("request_explaination(detected[Bonafide])")
    refs.append("Was Bonafide how the audio sample classified as?")
    refs.append("How was the audio classified as bonadide?")

    questions.append("request(detected[Bonafide])")
    questions.append("request_explaination(detected[Bonafide])")
    refs.append("Was the audio sample found to be a bonafide sample?")
    refs.append("How was the audio sample found to be a Bonafide sample?")

    questions.append("request(detected[Bonafide])")
    questions.append("request_explaination(detected[Bonafide])")
    refs.append("Was the audio sample classified as a bonafide sample?")
    refs.append("How was the audio sample classified as a Bonafide sample?")


    for i in arg:
        questions.append("request(detected[Bonafide], detected_by[{}], shap_value[{:.3f}])".format(i[0], i[1]))
        questions.append("request_explaination(detected[Bonafide], detected_by[{}], shap_value[{:.3f}])".format(i[0], i[1]))
        refs.append("was the Bonafide audio sample detected by {}, with a shap value of {:.3f}?".format(i[0], i[1]))
        refs.append("How was {}, with a shap value of {:.3f}, determined to impact the classification of the audio sample as bonafide?".format(i[0], i[1]))

        questions.append("request(detected[Bonafide], detected_by[{}], shap_value[{:.3f}])".format(i[0], i[1]))
        questions.append("request_explaination(detected[Bonafide], detected_by[{}], shap_value[{:.3f}])".format(i[0], i[1]))
        refs.append("was the audio sample detected as bonafide by {}, with a shap value of {:.3f}?".format(i[0], i[1]))
        refs.append("How was the audio sample detected as bonafide by {}, with a shap value of {:.3f}?".format(i[0], i[1]))

        questions.append("request(detected[Bonafide], detected_by[{}], shap_value[{:.3f}])".format(i[0], i[1]))
        questions.append("request_explaination(detected[Bonafide], detected_by[{}], shap_value[{:.3f}])".format(i[0], i[1]))
        refs.append("was the audio sample found to be a bonafide sample by {}, with a shap value of {:.3f}?".format(i[0], i[1]))
        refs.append("How was the audio sample found to be a bonafide sample by {}, with a shap value of {:.3f}".format(i[0], i[1]))

        questions.append("request(detected[Bonafide], detected_by[{}], shap_value[{:.3f}])".format(i[0], i[1]))
        questions.append("request_explaination(detected[Bonafide], detected_by[{}], shap_value[{:.3f}])".format(i[0], i[1]))
        refs.append("was the audio sample classified as a bonafide sample by {}, with a shap value of {:.3f}?".format(i[0], i[1]))
        refs.append("How was the audio sample classified as a bonafide sample by {}, with a shap value of {:.3f}".format(i[0], i[1]))


    return questions, refs

def reverberation(arg):
    questions = []
    refs = []
    questions.append("request(detected_signature[Reverberation])")
    questions.append("request_explanation(detected_signature[Reverberation])")
    questions.append("request(detected_signature[Reverberation]), number_of_signatures[Single_Reverberation]")
    questions.append("request_explanation(detected_signature[Reverberation]), number_of_signatures[Single_Reverberation]")
    questions.append("request(detected_signature[Reverberation]), number_of_signatures[Multiple_Reverberation]")
    questions.append("request_explanation(detected_signature[Reverberation]), number_of_signatures[Multiple_Reverberation]")

    refs.append("Was the audio sample detected as having Reverberation signature(s)?")
    refs.append("How was the Reverberation signature detected in the audio sample?")
    refs.append("Was there a single Reverberation signature detected in the audio sample?")
    refs.append("How was the single Reverberation signature detected in the audio sample?")
    refs.append("Was there multiple Reverberation signatures detected in the audio sample?")
    refs.append("How were the multiple Reverberation signatures detected in the audio sample?")

    for i in arg:
        questions.append("request(detected_signature[Reverberation], detected_by[{}])".format(i))
        questions.append("verify_attribute(detected_signature[Reverberation], detected_by[{}])".format(i))

        refs.append("Was the Reverberation signature in the audio file was detected by {}?".format(i))
        refs.append("The Reverberation signature in the audio sample was detected by {}, was "
                    "the Reverberation signature detected by other features?".format(i))

    return questions, refs


def microphone(arg):
    questions = []
    refs = []
    questions.append("request(detected_signature[Microphone])")
    questions.append("request_explanation(detected_signature[Microphone])")
    questions.append("request(detected_signature[Microphone], number_of_signatures[Single_Microphone])")
    questions.append("request_explanation(detected_signature[Microphone], number_of_signatures[Single_Microphone])")
    questions.append("request(detected_signature[Microphone], number_of_signatures[Multiple_Microphone])")
    questions.append("request_explanation(detected_signature[Microphone], number_of_signatures[Multiple_Microphone])")

    refs.append("Was the audio sample detected as having a microphone signature?")
    refs.append("How was the Microphone signature detected in the audio sample?")
    refs.append("Was there a single Microphone signature detected in the audio sample?")
    refs.append("How was the single Microphone signature detected in the audio sample?")
    refs.append("Was there multiple microphone signatures detected in the audio sample?")
    refs.append("How were the multiple Microphone signatures detected in the audio sample?")

    for i in arg:
        questions.append("request(detected_signature[Microphone], detected_by[{}])".format(i))
        questions.append("verify_attribute(detected_signature[Microphone], detected_by[{}])".format(i))

        refs.append("Was the Microphone signature in the audio file was detected by {}?".format(i))
        refs.append("The Microphone signature in the audio sample was detected by {}, "
                    "was the Microphone signature detected by other features?".format(i))

    return questions, refs


def get_important_feats(vals, cols, num) -> (list):
    shap_vals = vals
    shap_vals = pd.DataFrame(shap_vals, columns=cols) #['mfcc-{}'.format(i) for i in range(shap_vals.shape[1])])
    means = pd.DataFrame(shap_vals.abs().mean(axis=0))
    means_sorted = means.sort_values(by=[0], ascending=False)

    out = list(means_sorted.index[:num])

    return out


def load_graph(graph_dir):
    import rdflib.graph as g
    graph = g.Graph()
    graph.parse(graph_dir, format='application/rdf+xml')
    return graph


def get_level(node_name):
    query = """
                prefix afo: <https://w3id.org/afo/onto/1.1#>
                prefix owl:	<http://www.w3.org/2002/07/owl#>
                SELECT DISTINCT ?pred ?obj
                WHERE
                    {
                        ?sub ?pred ?obj .
                        """
    temp = "afo:{} rdfs:label ?obj .".format(node_name)
    try:
        result = list(graph.query(query + temp + " }"))
        return int(result[0][1].split('_')[-1])
    except:
        return None


def get_info(graph_obj, leaf_node, stop_level):
    leaf_lvl = get_level(leaf_node)
    if not leaf_lvl or leaf_lvl <= stop_level:
        return None

    out = OrderedDict()
    graph = graph_obj
    temp_lst = [leaf_node]

    #print(leaf_lvl, leaf_node)
    target = temp_lst.pop(0)
    query = """
            prefix afo: <https://w3id.org/afo/onto/1.1#>
            prefix owl:	<http://www.w3.org/2002/07/owl#>
            SELECT DISTINCT ?pred ?obj
            WHERE
                {
                    ?sub ?pred ?obj .
                    """
    temp = "afo:{} ?pred ?obj .".format(target)
    result = graph.query(query + temp + " }")
    for i in result:
#        print(i)
        pred = i[0].split('#')[1]
        if pred != "label" and pred != "type":
            try:
                obj = str(i[1].split('#')[1])
            except:
                obj = str(i[1])
            if pred not in out:
                out[pred] = [obj]
            else:
                out[pred].append(obj)
    return out


def get_loc(node):
   # print(node)
    query = """
    prefix afo: <https://w3id.org/afo/onto/1.1#>
    prefix owl: <http://www.w3.org/2002/07/owl#>
    select ?subclass (count(?intermediate)-1 as ?depth) where {
    ?subclass rdfs:subClassOf* ?intermediate .
    ?intermediate rdfs:subClassOf*
    """
    temp = "afo:{} . ".format(node) + "}" + "group by ?subclass order by ?depth"
    result = graph.query(query + temp)
    dist = []
    for i in result:
        dist.append(i[1])
    return max(dist)



def get_individual(graph_obj, leaf_node, stop_level):
    if int(get_loc(leaf_node)) > stop_level:
        return None
    #print(leaf_node, get_loc(leaf_node), stop_level)
    out = OrderedDict()
    graph = graph_obj
    temp_lst = [leaf_node]


    target = temp_lst.pop(0)
    level = get_level(target) if get_level(target) is not None and str(get_level(target)).isalpha() else ""
    query = """
            prefix afo: <https://w3id.org/afo/onto/1.1#>
            prefix owl: <http://www.w3.org/2002/07/owl#>
            prefix dbo: <http://dbpedia.org/resource/>
            SELECT DISTINCT ?sub ?pred ?obj
            WHERE
                {
                  ?sub ?pred ?obj .
                    """

    temp = "?sub rdf:type afo:{} .".format(target)
    result = graph.query(query + temp + " }")
    #print(query + temp + " }")
    for i in result:
        pred = i[1].split('#')[-1] if "#" in i[1] else i[1].split("/")[-1]
        if pred != "label":
            try:
                obj = str(i[2].split('#')[1])
            except:
                obj = str(i[2])
            if pred not in out:
                out[pred] = [obj]
            else:
                out[pred].append(obj)
    return out





def enrich(graph, sentences, mr, level):
    from nltk.corpus import stopwords
    s = set(stopwords.words('english'))
    new_mr = mr[:-1].split(",")
    out_sentences = []
    sentences = [sentences]
    for sentence in sentences:
        added = []
        t = sentence.translate(str.maketrans('', '', "’" + string.punctuation.replace("-", "").replace(".", ""))).split()
        last_word = t.pop()
        t.append(last_word.translate(str.maketrans('', '', string.punctuation.replace("-", "").replace(".", ""))))
        new_t = t[:]
        for i in t:
            if i[0] == "-" or '.' in i or i in s:
                continue
            new_i = i[0].upper() + i[1:]
            individual = get_individual(graph, new_i, level)
            indx = new_t.index(i)
            new_t[indx].replace(" ", "")
            try:
                new_t[indx] = individual["subject"][0].split(':')[-1]
                new_mr[-1].replace(i, individual["subject"][-1].split(':')[-1])
                print("individual >> ", individual["subject"])
            except:
                pass
            knowledge = get_info(graph, new_i, level)
            if knowledge != None:
                temp = ""
                indx = new_t.index(i)
                new_t[indx].replace(" ", "")
                if "subClassOf" in knowledge and not individual:
                    print("b4",new_t)
                    new_t[indx] = knowledge["subClassOf"][-1]
                    print("after",new_t)
                    new_mr[-1].replace(i, knowledge["subClassOf"][-1])
        #            new_temp = ", a {}, ".format(knowledge["subClassOf"][0])
        #            attrib = " subClassOf[{}]".format(knowledge["subClassOf"][0])
                    #if attrib not in new_mr: new_mr.append(attrib)
                    #if knowledge["subClassOf"][0] not in t: temp += new_temp
                    #del knowledge["subClassOf"]

                if "minimum" in knowledge:
                    new_temp ="having minimum value of {}, ".format(knowledge["minimum"][0])
                    attrib = " minimum[{}]".format(knowledge["minimum"][0])
                    if attrib not in new_mr: new_mr.append(attrib)
                    if knowledge["minimum"][0] not in t: temp += new_temp
                    del knowledge["minimum"]

                if "maximum" in knowledge:
                    new_temp = "and a maximum value of {}, ".format(knowledge["maximum"][0])
                    attrib = " maximum[{}]".format(knowledge["maximum"][0])
                    if attrib not in new_mr: new_mr.append(attrib)
                    if knowledge["maximum"][0] not in t: temp += new_temp
                    del knowledge["maximum"]
                '''
                if "captured_by" in knowledge:
                    if "detected_by" in mr:
                        temp += " was captured by {}, ".format(knowledge["captured_by"][  ])
                        attrib = " captured_by[{}]".format(knowledge["captured_by"][  ])
                    else:
                        temp += " was captured by {}, ".format(", ".join(knowledge["captured_by"]))
                        attrib = " captured_by[{}]".format(", ".join(knowledge["captured_by"]))
                    if attrib not in new_mr: new_mr.append(attrib)
                    del knowledge["captured_by"]
                '''
                #if "detected_by" in knowledge:
                if "Replayed" in t or "replayed" in t or "spoofed" in t or "Spoofed" in t:
                    new_temp = "detected by {}, ".format("CNN" )
                    attrib = " detected_by[{}]".format("CNN")
                    if attrib not in new_mr: new_mr.append(attrib)
                    if "CNN" not in t and new_temp not in temp: temp += new_temp
  
                elif "speaker" in sentence or "Speaker" in sentence:
                    new_temp = "detected by {}, ".format("SVM" )
                    attrib = " detected_by[{}]".format("SVM" )
                    if attrib not in new_mr: new_mr.append(attrib)
                    if "SVM" not in t and new_temp not in temp: temp += new_temp

                #else:
                #    new_temp = "was detected by {}, ".format(knowledge["detected_by"][0])
                #    attrib = " detected_by[{}]".format(knowledge["detected_by"][0])
          #      if attrib not in new_mr: new_mr.append(attrib)
          #      if "CNN" not in t or "SVM" not in t: temp += new_temp
                #del knowledge["detected_by"]

                if "number_of_coefficients" in knowledge:
                    new_temp = "has {} coefficients, ".format(knowledge["number_of_coefficients"][0])
                    attrib = " number_of_coefficients[{}]".format(knowledge["number_of_coefficients"][0])
                    if attrib not in new_mr: new_mr.append(attrib)
                    if new_temp not in sentence or new_temp not in t: temp += new_temp
                    del knowledge["number_of_coefficients"]

                if "informs_about" in knowledge:
                    new_temp = "informs about {}, ".format(knowledge["informs_about"][0])
                    attrib = " informs_about[{}]".format(knowledge["informs_about"][0])
                    if attrib not in new_mr: new_mr.append(attrib)
                    if new_temp not in sentence or new_temp not in t: temp += new_temp
                    del knowledge["informs_about"]

                if "derivative_of" in knowledge:
                    new_temp = "is a derivative of {}, ".format(knowledge["derivative_of"][0])
                    attrib = " derivative_of[{}]".format(knowledge["derivative_of"][0])
                    if attrib not in new_mr: new_mr.append(attrib)
                    if knowledge["derivative_of"][0] not in t: temp += new_temp
                    del knowledge["derivative_of"]

                if "second_derivative_of" in knowledge:
                    new_temp = "is a second derivative of {}, ".format(knowledge["second_derivative_of"][0])
                    attrib = " second_derivative_of[{}]".format(knowledge["second_derivative_of"][0])
                    if attrib not in new_mr: new_mr.append(attrib)
                    if knowledge["second_derivative_of"][0] not in t: temp += new_temp
                    del knowledge["second_derivative_of"]

                if "has_operation" in knowledge:
                    new_temp = "calculated using {}, ".format(", ".join(knowledge["has_operation"]))
                    attrib = " has_operation[{}]".format(", ".join(knowledge["has_operation"]))
                    if attrib not in new_mr: new_mr.append(attrib)
                    if knowledge["has_operation"] not in t: temp += new_temp
                    del knowledge["has_operation"]

                if "uses_filterbank" in knowledge:
                    new_temp = "uses {}, ".format(knowledge["uses_filterbank"][0])
                    attrib = " uses_filterbank[{}]".format(knowledge["uses_filterbank"][0])
                    if attrib not in new_mr: new_mr.append(attrib)
                    if knowledge["uses_filterbank"][0] not in t: temp += new_temp
                    del knowledge["uses_filterbank"]

                if "number_of_delta_coefficients" in knowledge:
                    new_temp = "has {} delta coefficients, ".format(knowledge["number_of_delta_coefficients"][0])
                    attrib = " number_of_delta_coefficients[{}]".format(knowledge["number_of_delta_coefficients"][0])
                    if attrib not in new_mr: new_mr.append(attrib)
                    if new_temp not in sentence or new_temp not in t: temp += new_temp
                    del knowledge["number_of_delta_coefficients"]

                if "number_of_delta_delta_coefficients" in knowledge:
                    attrib = " number_of_delta_delta_coefficients[{}]".format(knowledge["number_of_delta_delta_coefficients"][0])
                    new_temp = "has {} delta delta coefficients, ".format(knowledge["number_of_delta_delta_coefficients"][0])
                    if attrib not in new_mr: new_mr.append(attrib)
                    if new_temp not in sentence or new_temp not in t: temp += new_temp
                    del knowledge["number_of_delta_delta_coefficients"]

                new_t.insert(indx+1, temp)
        out_sentences.append(" ".join(new_t)+"?")
    new_mr = ",".join(new_mr) +  ")"
    return out_sentences, new_mr


def need_enrich(_comp_dir, offset=0):
    comp_data = pd.read_csv(_comp_dir)

    #  CHANGE BELOW TO THE OUTPUT OF THE TRAINED UNENRICH SENTENCE LEVEL SCORES

    if _comp_dir.split('_')[-1].split('.')[0] == "FC":
        unenrich_data = pd.read_csv('../evaluations/sent_scores_{}_output2_frink_partial_enrich_-1_DataTuner_No_FC.csv'.format(PREFIX) )
    else:
        unenrich_data = pd.read_csv('../evaluations/sent_scores_{}_output2_frink_partial_enrich_-1_DataTuner_No_FC_No_FS.csv'.format(PREFIX) )

    base_data = unenrich_data

    threshold = np.mean(unenrich_data["bleu"].to_numpy())
    print("Threshold (AVG sentence blue for non-enriched): {}\nOffset set to: {}\n".format(threshold, offset))
    if base_data.shape[0] > comp_data.shape[0]:
        base_data = base_data.iloc[:comp_data.shape[0], :]
    elif base_data.shape[0] < comp_data.shape[0]:
        comp_data = comp_data.iloc[:base_data.shape[0], :]

    accept = []
    for indx in range(base_data.shape[0]):
        diff = comp_data["bleu"].iloc[indx] - base_data["bleu"].iloc[indx]
        if diff > 0 and comp_data["bleu"].iloc[indx] < threshold+offset:
            accept.append(indx)
    print("Length of selected for enrichment: {}".format(len(accept)))
    return accept



if __name__ == "__main__":
    graph = load_graph("audioFeatVocabXML3.owl")

    #  CHANGE DEPENDING ON NEXT ENRICHMENT LEVEL AND PREFIX ATTACHED TO FILES
    PREFIX = "bf"  #"bk" "jj" #"TEST2"
    ENRICH_LVL = 0
    offset = 0.05
    if PREFIX == "bk":
        already = [38, 134, 135, 143, 144, 149, 162, 182, 191, 193, 194, 199]
       #already = [29, 30, 33, 35, 53]
    elif PREFIX == "jj":
        already = [4, 8, 12, 16, 18, 19, 20, 27, 30, 33, 35, 36, 41, 44, 48, 59, 61, 65]
       #already = [4, 20, 150, 151, 153, 166, 168, 182, 189]
    #prev_scores_dir = '../evaluations/sent_scores_{}_output2_frink_partial_enrich_{}_DataTuner_No_FC.csv'.format(PREFIX, ENRICH_LVL-2)
    #current_scores_dir = '../evaluations/sent_scores_{}3_output2_frink_partial_enrich_{}_DataTuner_No_FC.csv'.format(PREFIX, ENRICH_LVL-1)
    #model = prev_scores_dir.split("_")[-1].split('.')[0]
#    print('>>>>>>>>>>>>>>>', prev_scores_dir.split("_")[-1].split('.')[0])

    if ENRICH_LVL > -1:
       
        #accept_indxs = need_enrich(current_scores_dir, offset)
        #print(">>>", len(accept_indxs))


        for NAME in ['train', 'test', 'valid']:
            print("NAME >>", NAME)
            if ENRICH_LVL == 1:
                try:
                    prev_questions = pd.read_csv("{}3_viggo_audio_enrich_{}_{}_{}.csv".format(PREFIX, ENRICH_LVL-2, NAME, model))
                except:
                    prev_questions = pd.read_csv("{}3_viggo_audio_enrich_{}_{}.csv".format(PREFIX, ENRICH_LVL-2, NAME))
                prev_p = prev_questions["mr"].to_numpy()
                prev_p2 = prev_questions["ref"].to_numpy()
                prev_p3 = zip(prev_p, prev_p2)
                prev_p3 = list(prev_p3)
                print("len prev p {} len prev p2 {}".format(len(prev_p), len(prev_p2)))
                prev_dups = []
                print("len prev P3 {}".format(len(prev_p3)))

            if ENRICH_LVL > 0:
                questions_in = pd.read_csv("{}3_viggo_audio_enrich_{}_{}_{}.csv".format(PREFIX, ENRICH_LVL-1, NAME, model))
                print("Q in dir:", "{}3_viggo_audio_enrich_{}_{}_{}.csv".format(PREFIX, ENRICH_LVL-1, NAME, model))
            else:
                questions_in = pd.read_csv("enriched_dtuner/{}_viggo_audio_enrich_{}_{}.csv".format(PREFIX, ENRICH_LVL-1, NAME))
                print("Q in dir:", "{}_viggo_audio_enrich_{}_{}.csv".format(PREFIX, ENRICH_LVL-1, NAME))

            questions = OrderedDict()
            enrich_questions = OrderedDict()

            p = questions_in["mr"].tolist()
            p2 = questions_in["ref"].tolist()
            print("len P {} len P2 {}".format(len(p), len(p2)))
            p3 = zip(p, p2)
            p3 = list(p3)
            print("len P3 {}".format(len(p3)))

            for cnt, item in enumerate(p3):
                command = item[0]
                ref = item[1]
                if ENRICH_LVL == 1:
#                    print(">>>", cnt, len(p3))
                    if cnt not in accept_indxs:
                        if prev_p3[cnt][0] not in questions:
                            questions[prev_p3[cnt][0]] = [prev_p3[cnt][1]]
                        else:
                            questions[prev_p3[cnt][0]].append(prev_p3[cnt][1])
                    else:
                        print("!!!!!! YES", cnt)
                        ref_temp, command_temp  = enrich(graph, prev_p3[cnt][1], prev_p3[cnt][0], ENRICH_LVL)
                        if command_temp not in questions:
                            questions[command_temp] = [ref_temp]
                        else:
                            questions[command_temp].append(ref_temp)

                elif ENRICH_LVL == 0 or (cnt in accept_indxs and cnt in already):
                    print("!!!!!! YES", cnt)
                    print("command >>>>>>", command)
                    ref_temp, command_temp  = enrich(graph, ref, command, ENRICH_LVL)
                    if command_temp not in questions:
                        questions[command_temp] = [ref_temp[0]]
                    else:
                        questions[command_temp].append(ref_temp[0])
                    print("question >>>>>", command_temp, len(questions[command_temp]))
                
                else: #ENRICH_LVL > 1 and cnt not in accept_indxs:
                    print("not", cnt)
                    if command not in questions:
                        questions[command] = [ref]
                    else:
                        questions[command].append(ref)

                #else:
                 #   print("\n>>>>> Hit end of line\n")
#                       print(ref_temp)
            if ENRICH_LVL > 0: print("questions out {} selected {}".format(len(questions), len(accept_indxs)))

            out = []
            for i in questions:
                print("out refs len {}".format(len(questions[i])))
                for ref in questions[i]:
                    if type(ref) is list: ref=ref[0]
                    out.append((i,ref))
            print("len out {}".format(len(out)))
            out_df = pd.DataFrame(out, columns=['mr', 'ref'])
            out_df.to_csv("{}_viggo_audio_enrich_{}_{}.csv".format(PREFIX, ENRICH_LVL, NAME), index=False) #{}.csv".format(PREFIX, ENRICH_LVL, NAME, model), index=False)

    else:
        questions = OrderedDict()
        label = ["Replayed", "Bonafide", "1", "2", "3", "4", "5", "6", "7", "8"]
        imp_feats = []
        for i in range(13):
            imp_feats.append(("PSRCC-{}".format(i), round(np.random.uniform(-3, 3, 1)[0], 4)))
            imp_feats.append(("MSRCC-{}".format(i), round(np.random.uniform(-3, 3, 1)[0], 4)))
        for i in range(1, 14):
            imp_feats.append(("GTCC-{}".format(i), round(np.random.uniform(-3, 3, 1)[0]), 4))
        for i in range(42):
            imp_feats.append(("MFCC-{}".format(i), round(np.random.uniform(-3, 3, 1)[0], 4)))
        for i in range(60):
            imp_feats.append(("LFCC-{}".format(i), round(np.random.uniform(-3, 3, 1)[0], 4)))
        p = []
        p2 = []
        for y in label:
            mr, ref = detection_switch(y, imp_feats)
            p += mr
            p2 += ref

        for cnt, command in enumerate(p):
            if command not in questions:
                print(">>>>>", command)
                questions[command] = [p2[cnt]]
            else:
                questions[command].append(p2[cnt])

      
        q_train = {"mr":[], "ref":[]}
        q_test = {"mr":[], "ref":[]}
        q_val = {"mr":[], "ref":[]}
        for indx, i in enumerate(questions):
            split_num = math.floor(0.25*len(questions[i]))
            if split_num <= 0:
                train_out, test_out = train_test_split(questions[i], test_size=math.ceil(0.25*len(questions[i])), random_state=0)
            else:
                train_out, test_out =  train_test_split(questions[i], test_size=split_num, random_state=0)
            train_out, val_out = train_test_split(train_out, test_size=len(test_out), random_state=0)
 
            for text in train_out:
                q_train['mr'].append(i)
                q_train['ref'].append(text)

            for text in test_out:
                q_test['mr'].append(i)
                q_test['ref'].append(text)

            for text in val_out:
                q_val['mr'].append(i)
                q_val['ref'].append(text)

        q_train2 = pd.DataFrame(q_train)
        q_test2 = pd.DataFrame(q_test)
        q_val2 = pd.DataFrame(q_val)

        q_train2.to_csv("TEST2_viggo_audio_enrich_{}_train.csv".format(ENRICH_LVL), index=None)
        q_test2.to_csv("TEST2_viggo_audio_enrich_{}_test.csv".format(ENRICH_LVL), index=None)
        q_val2.to_csv("TEST2_viggo_audio_enrich_{}_val.csv".format(ENRICH_LVL), index=None)
    
