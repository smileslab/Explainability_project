import sys
import json
from multiprocessing import Process, Pool, Array, cpu_count
from mpi4py import MPI


import pickle
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import string
import math
from tqdm import tqdm
from collections import OrderedDict, Counter

sys.path.append(os.getcwd())

def load_graph(graph_dir):
    import rdflib.graph as g
    graph = g.Graph()
    graph.parse(graph_dir, format='application/rdf+xml')
    print("GRAPH LOADED!!")
    return graph


def get_label(node_name):
    query = """
                prefix afo: <https://w3id.org/afo/onto/1.1#>
                prefix owl:     <http://www.w3.org/2002/07/owl#>
                SELECT DISTINCT ?pred ?obj
                WHERE
                    {
                        ?sub ?pred ?obj .
                        """
    temp = "afo:{} rdfs:label ?obj .".format(node_name)
    try:
        result = list(graph.query(query + temp + " }"))
        return result[0][1]
    except:
        return None

'''
def get_individual(graph_obj, leaf_node, stop_level):
    if leaf_node not in LEVELS:
        LEVELS[leaf_node] = get_loc(leaf_node)
    if int(LEVELS[leaf_node]) > stop_level:
        return None
    #print(leaf_node, get_loc(leaf_node), stop_level)
    out = OrderedDict()
    graph = graph_obj
    temp_lst = [leaf_node]


    target = temp_lst.pop(0)
    print(">>>>> Target >>>>", target)
    if(leaf_node in LEVELS):
        level = LEVELS[target]

    query = """
            prefix afo: <https://w3id.org/afo/onto/1.1#>
            prefix owl:	<http://www.w3.org/2002/07/owl#>
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
'''

############################################# VVVVV NEW VVVVV ################################


def get_attribs(graph_obj, obj):
    graph = graph_obj
    query = """
                    prefix afo: <https://w3id.org/afo/onto/1.1#>
                    prefix owl: <http://www.w3.org/2002/07/owl#>
                    prefix dbo: <http://dbpedia.org/resource/>
                    SELECT DISTINCT ?sub ?pred ?obj
                    WHERE
                        {
                            ?sub ?pred ?obj .
                            """

    temp1 = "{" + "<https://w3id.org/afo/onto/1.1#{}> ?pred ?obj . FILTER(regex(str(?sub), \"{}\" ) )".format(obj, obj) + "}" + "UNION" + "{" + "<http://dbpedia.org/resource/{}> ?pred ?obj . FILTER(regex(str(?sub), \"{}\" ) )".format(obj, obj) + "}"
 #   print(query + temp1 + "}")
    result = graph.query(query + temp1 + "}")
    if not result:
        query = """
                            prefix afo: <https://w3id.org/afo/onto/1.1#>
                            prefix owl: <http://www.w3.org/2002/07/owl#>
                            prefix dbo: <http://dbpedia.org/resource/>
                            SELECT DISTINCT ?sub ?pred ?obj
                            WHERE
                                {
                                    ?sub ?pred ?obj .
                                    """

        temp1 = "?sub rdfs:label \"{}\"@en .".format(obj)
        temp2 = "?sub rdfs:label \"{}\" .".format(obj)
      #  print(query + "{" + temp1 + " }" + " UNION " + "{" + temp2 + " }" + "}")
        result = graph.query(query + "{" + temp1 + " }" + " UNION " + "{" + temp2 + " }" + "}")

    return result


def get_same(graph_obj, obj):
    graph = graph_obj
    query = """
                        prefix afo: <https://w3id.org/afo/onto/1.1#>
                        prefix owl: <http://www.w3.org/2002/07/owl#>
                        prefix dbo: <http://dbpedia.org/resource/>
                        SELECT DISTINCT ?sub2
                        WHERE
                            {
                                ?sub ?pred ?obj .
                                """

    temp1 = "?sub rdfs:label \"{}\"@en .".format(obj)
    temp2 = "?sub rdfs:label \"{}\" .".format(obj)
    temp3 = "?sub2 owl:sameAs ?sub ."
    #print(query + "{" + temp1 + " }" + " UNION " + "{" + temp2 + " }" + "}")
    result = graph.query(query + "{" + temp1 + " }" + " UNION " + "{" + temp2 + " }" + temp3 + "}")
    return result


def get_all_attribs(graph_obj, obj, stop_level):
    if obj in LEVELS and int(LEVELS[obj]['level']) >= stop_level:
        attribs = {}
        data0 = get_attribs(graph, obj)
        for i in data0:
            pred = i[1].split('#')[-1]
            obj = i[2].split('#')[-1] if '#' in i[2] else i[2]
            if pred not in attribs:
                attribs[pred] = [obj]
            else:
                if obj not in attribs[pred]:
                    attribs[pred].append(obj)

        data2 = get_same(graph, obj)
        for i in data2:
            data = get_attribs(graph, i[0])
            for j in data:
                pred = j[1].split('#')[-1]
                obj = j[2].split('#')[-1] if '#' in j[2] else j[2]
                if pred not in attribs:
                    attribs[pred] = obj
                else:
                    if obj not in attribs[pred]:
                        attribs[pred].append(obj)
        return attribs
    else:
        return None
################################ ^^^^ NEW ^^^^ ############################

'''
def get_info(graph_obj, leaf_node, stop_level):
    if leaf_node not in LEVELS:
        LEVELS[leaf_node] = get_loc(leaf_node)
    if int(LEVELS[leaf_node]) > stop_level:
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
            SELECT DISTINCT ?sub ?pred ?obj
            WHERE
                {
                  ?sub ?pred ?obj .
                    """

    temp = " afo:{} ?pred ?obj .".format(target)
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
            elif obj not in out[pred]:
                out[pred].append(obj)
    return out
'''


def get_loc(node):
    query = """
    prefix afo: <https://w3id.org/afo/onto/1.1#>
    prefix owl:	<http://www.w3.org/2002/07/owl#>
    prefix dbo: <http://dbpedia.org/resource/>
    select ?subclass (count(?intermediate)-1 as ?depth) 
    where {
        ?subclass rdfs:subClassOf* ?intermediate .
    """
    temp = "?intermediate rdfs:subClassOf* <https://w3id.org/afo/onto/1.1#{}> .".format(node) + "} group by ?subclass order by ?depth"
    result = graph.query(query + temp)
    dist = []
    for i in result:
        dist.append(i[1])
    #print(">>>>>>", node, max(dist))
    return max(dist)


def enrich(sentences, mr):
    level = ENRICH_LVL
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
           # individual = get_individual(graph, new_i, level)
           # print("!!!!!!!!!!!!", individual)
            indx = new_t.index(i)
            new_t[indx].replace(" ", "")
            #if True:
           #     try:
           #         new_t[indx] = individual["subject"][0].split(':')[-1]
           #         new_mr[-1].replace(i, individual["subject"][-1].split(':')[-1])
           #     except:
           #         pass
            #print("!!!!!!!!!!!!", new_i)
            knowledge = get_all_attribs(graph, new_i, level)
            #print(new_i, knowledge)
            if knowledge != None:
                temp = ""
                if "subClassOf" in knowledge and knowledge["subClassOf"][-1] not in new_t :
                    #print("b4",new_t)
                    new_t[indx] = knowledge["subClassOf"][-1]
                    #print("after",new_t)
                    for indx, z in enumerate(new_mr):
                        if  new_i in z:
                            new_mr[indx] = new_mr[indx].replace(new_i, knowledge["subClassOf"][-1])
#                   if i in new_mr:
                               #print(new_mr[-1].index(i), new_mr[-1][new_mr[-1].index(i) : new_mr[-1].index(i)+len(i)], knowledge["subClassOf"])

                if True: #level == 0:
                    
                    if "minimum" in knowledge:
                        new_temp = ",having minimum value of {},".format(knowledge["minimum"][0])
                        attrib = " minimum[{}]".format(knowledge["minimum"][0])
                        if attrib not in new_mr: new_mr.append(attrib)
                        if new_temp not in sentence and new_temp not in t: temp += new_temp

                    if "maximum" in knowledge:
                        new_temp = ",and a maximum value of {},".format(knowledge["maximum"][0])
                        attrib = " maximum[{}]".format(knowledge["maximum"][0])
                        if attrib not in new_mr: new_mr.append(attrib)
                        if new_temp not in sentence and new_temp not in t: temp += new_temp
                
                  #  if "captured_by" in knowledge:
                  #      if len(set(knowledge["captured_by"]).intersection(t)) == 0:
                  #          new_temp = ",captured by {},".format(", ".join(knowledge["captured_by"]))

                  #          if new_temp not in sentence and new_temp not in t: temp += new_temp
             
                    if new_i in ["Replayed", "replayed", "Spoofed", "spoofed"]: #if "Replayed" in t or "replayed" in t or "spoofed" in t or "Spoofed" in t:
                        new_temp = "was detected by {}, ".format("CNN" )
                        attrib = " detected_by[{}]".format("CNN")
                            
                        if attrib not in new_mr: new_mr.append(attrib)
                        if "CNN" not in t and new_temp not in new_t: temp += new_temp

                    if new_i in ["Speaker", "speaker"]: #elif "speaker" in sentence or "Speaker" in sentence:
                        new_temp = "was detected by {}, ".format("SVM" )
                        attrib = " detected_by[{}]".format("SVM" )

                        if attrib not in new_mr: new_mr.append(attrib)
                        if "SVM" not in t and new_temp not in new_t: temp += new_temp
              
                   
                    if "number_of_coefficients" in knowledge:
                        new_temp = ", has {} coefficients,".format(knowledge["number_of_coefficients"][0])
                        attrib = " number_of_coefficients[{}]".format(knowledge["number_of_coefficients"][0])
                        if attrib not in new_mr: new_mr.append(attrib)
                        if new_temp not in sentence and new_temp not in t: temp += new_temp

                    if "informs_about" in knowledge:
                        new_temp = ", informs about {},".format(knowledge["informs_about"][0])
                        attrib = " informs_about[{}]".format(knowledge["informs_about"][0])
                        if attrib not in new_mr: new_mr.append(attrib)
                        if new_temp not in sentence and new_temp not in t: temp += new_temp

                    if "derivative_of" in knowledge:
                        new_temp = "is a derivative of {},".format(knowledge["derivative_of"][0])
                        attrib = " derivative_of[{}]".format(knowledge["derivative_of"][0])
                        if attrib not in new_mr: new_mr.append(attrib)
                        if new_temp not in sentence and new_temp not in t: temp += new_temp

                    if "second_derivative_of" in knowledge:
                        new_temp = ", is a second derivative of {},".format(knowledge["second_derivative_of"][0])
                        attrib = " second_derivative_of[{}]".format(knowledge["second_derivative_of"][0])
                        if attrib not in new_mr: new_mr.append(attrib)
                        if new_temp not in sentence and new_temp not in t: temp += new_temp
        
                    if "has_operation" in knowledge:
                        new_temp = ", calculated using {},".format(", ".join(knowledge["has_operation"]))
                        attrib = " has_operation[{}]".format(", ".join(knowledge["has_operation"]))
                        if attrib not in new_mr: new_mr.append(attrib)
                        if new_temp not in sentence and new_temp not in t: temp += new_temp

                    if "uses_filterbank" in knowledge:
                        new_temp = ", uses {},".format(knowledge["uses_filterbank"][0])
                        attrib = " uses_filterbank[{}]".format(knowledge["uses_filterbank"][0])
                        if attrib not in new_mr: new_mr.append(attrib)
                        if new_temp not in sentence and new_temp not in t: temp += new_temp

                    if "number_of_delta_coefficients" in knowledge:
                        new_temp = ", has {} delta coefficients, ".format(knowledge["number_of_delta_coefficients"][0])
                        attrib = " number_of_delta_coefficients[{}]".format(knowledge["number_of_delta_coefficients"][0])
                        if attrib not in new_mr: new_mr.append(attrib)
                        if new_temp not in sentence and new_temp not in t: temp += new_temp

                    if "number_of_delta_delta_coefficients" in knowledge:
                        attrib = ", number_of_delta_delta_coefficients[{}]".format(knowledge["number_of_delta_delta_coefficients"][0])
                        new_temp = "has {} delta delta coefficients, ".format(knowledge["number_of_delta_delta_coefficients"][0])
                        if attrib not in new_mr: new_mr.append(attrib)
                        if new_temp not in sentence and new_temp not in t: temp += new_temp
   
                    new_t.insert(indx + 1, temp)
        new_t[:] = [x for x in new_t if x != '']
        if sentence[-1] in string.punctuation:
            out_sentences.append((" ".join(new_t) + sentence[-1]))
        else:
            out_sentences.append(" ".join(new_t))
    new_mr = ",".join(new_mr) + ")"
    #print(out_sentences, "\n")
    if len(out_sentences) == 1:
    	return new_mr, out_sentences[0]
    else:
        return new_mr, out_sentences


'''
def replace_word(graph, sentences, mr, level):
    from nltk.corpus import stopwords
    s = set(stopwords.words('english'))
    new_mr = mr[:-1]
    out_sentences = []
    sentences = [sentences]
    for sentence in sentences:
        added = []
        t = sentence.translate(str.maketrans('', '', string.punctuation.replace("-", "").replace(".", "")))
        t = t.split()
        last_word = t.pop()
        t.append(last_word.translate(str.maketrans('', '', string.punctuation)))
        new_t = t[:]
        for i in t:
            if i[0] == "-" or i in s:
                continue
            new_i = i[0].upper() + i[1:]
            knowledge = get_all_attribs(graph, new_i, level)
            if knowledge is not None:
                temp = ""
                indx = new_t.index(i)
                new_t[indx].replace(" ", "")

                if "subClassOf" in knowledge:
                    new_t[indx] = knowledge["subClassOf"][-1]
                    new_mr.replace(i, knowledge["subClassOf"][-1])
                    #print(knowledge)
                if "sameAs" in knowledge:
                    new_t[indx] = knowledge["sameAs"][-1]
    out_sentences = " ".join(new_t)
    return out_sentences, new_mr
'''


def format_clean(qa_lst, name):
    fp2 = open(_dir.split("\\")[-1].split("_{}".format(ENRICH_LVL-1))[0] + "_{}".format(ENRICH_LVL) +
                          _dir.split("\\")[-1].split("_{}".format(ENRICH_LVL-1))[1] + "_" + name, 'w')
    for i in range(len(qa_lst)):
        q = "Q: {}\n".format(qa_lst[i][0])
        a = "A: {}\n".format(qa_lst[i][1])
        temp = "===\n"
        for sent in [q, a, temp]:
            fp2.writelines(sent)

'''
def get_leaves(grph):
    leaves_lvls = {}
    query1 = """
    prefix afo: <https://w3id.org/afo/onto/1.1#>
    prefix owl:	<http://www.w3.org/2002/07/owl#>
    SELECT DISTINCT ?cls
    WHERE
    {?cls rdfs:subClassOf ?sup.
    FILTER NOT EXISTS
    {?sub rdfs:subClassOf ?cls FILTER(?sub != ?cls && ?sub != owl:Nothing )}}
                            """
    result = grph.query(query1)
    for i in result:
        node = i[0].split("#")[-1]
        node_lvl = get_loc(i[0].split("#")[-1])
        if node_lvl is not None:
            leaves_lvls[node] = node_lvl
            get_parent(grph, node)


def get_parent(grph, node):
    query1 = """
    prefix afo: <https://w3id.org/afo/onto/1.1#>
    prefix owl:	<http://www.w3.org/2002/07/owl#>
    SELECT ?superClass 
    WHERE 
    {
    """
    temp = "afo:{} rdfs:subClassOf ?superClass . ".format(node)
    result = grph.query(query1+temp+"}")
    for i in result:
        parent = i[0].split("#")[-1]
    return parent
'''

def get_all_class_lvls():
    query = """
    prefix afo: <https://w3id.org/afo/onto/1.1#>
    prefix owl:	<http://www.w3.org/2002/07/owl#>
    SELECT DISTINCT ?sub ?lab
    WHERE {
        ?subj ?pred ?obj .
        ?sub rdfs:label ?lab .
            }
    """
    result = graph.query(query)
    print("Getting Class Levels\n")
    j = 0
    for i in tqdm(result):
        name = i[0].split('#')[-1].split("/")[-1]
        label = i[1].split('#')[-1].split("/")[-1]
        if name not in LEVELS:
            temp = {"label": None, "level": None}
            LEVELS[name] = temp
            LEVELS[name]["label"] = label
            LEVELS[name]["level"] = int(get_loc(name)[0])

def startProcesses(_dir):
    p = pd.read_csv(_dir)

    questions = p["ref"].tolist()
    mrs = p["mr"].tolist()

    new_questions = ["" for i in range(len(questions))]
    TEMP =  mrs[:]
    _n_wrks = 3 
    vals = list(zip(questions, TEMP))
    with Pool(_n_wrks) as pool:
        res = pool.starmap(enrich, vals)
        pool.close()
        pool.join()

        qa1_df = pd.DataFrame(res, columns=["mr", "ref"])
    #if not os.path.exists("new_dtuner/dtuner/bf-viggo-enrich8-{}/".format(ENRICH_LVL)):
    #    os.makedirs("new_dtuner/dtuner/bf-viggo-enrich8-{}/".format(ENRICH_LVL), exist_ok=False)
    #qa1_df.to_csv("new_dtuner/dtuner/bf-viggo-enrich8-{}/".format(ENRICH_LVL)+_dir.split("/")[-1].split(".")[0] + ".csv", index=False)

    if not os.path.exists("new_dtuner/dtuner/bf-viggo-enrich-11key-{}/".format(ENRICH_LVL)):
        os.makedirs("new_dtuner/dtuner/bf-viggo-enrich-11key-{}/".format(ENRICH_LVL), exist_ok=False)
    qa1_df.to_csv("new_dtuner/dtuner/bf-viggo-enrich-11key-{}/".format(ENRICH_LVL)+_dir.split("/")[-1].split(".")[0] + ".csv", index=False)

#    if not os.path.exists("new_dtuner/dtuner/bf-viggo-answer-11key-{}/".format(ENRICH_LVL)):
#        os.makedirs("new_dtuner/dtuner/bf-viggo-answer-11key-{}/".format(ENRICH_LVL), exist_ok=False)
#    qa1_df.to_csv("new_dtuner/dtuner/bf-viggo-answer-11key-{}/".format(ENRICH_LVL)+_dir.split("/")[-1].split(".")[0] + ".csv", index=False)



    '''
    for cnt, quest in enumerate(tqdm(questions)):
        TEMP = mrs[cnt]
        q, new_mr = enrich(graph, quest, TEMP, ENRICH_LVL)
        new_questions[cnt] = q[0]
        mrs[cnt] = new_mr
    qa1_lst = list(zip(mrs, new_questions)) #, new_answers))
    qa1_df = pd.DataFrame(qa1_lst, columns=["mr", "ref"]) #, "answers"])


    if not os.path.exists("new_dtuner/dtuner/bf-viggo-enrich4-{}/".format(ENRICH_LVL)):
        os.makedirs("new_dtuner/dtuner/bf-viggo-enrich4-{}/".format(ENRICH_LVL), exist_ok=False)
    qa1_df.to_csv("new_dtuner/dtuner/bf-viggo-enrich4-{}/".format(ENRICH_LVL)+_dir.split("/")[-1].split(".")[0] + ".csv", index=False)
    '''


if __name__ == "__main__":
    LEVELS = {}

    graph = load_graph("audioFeatVocabXML4.owl")
    if not os.path.exists("./levels.json"):
        get_all_class_lvls()
        levels_json = json.dumps(LEVELS)
        fp = open('levels.json', 'w')
        fp.write(levels_json)
        fp.close()
    else:
        fp = open('levels.json', 'r')
        LEVELS = json.load(fp)

   #  CHANGE DEPENDING ON NEXT ENRICHMENT LEVEL AND PREFIX ATTACHED TO FILES
    #ENRICH_LVL = 3
    for ENRICH_LVL in range(0,3):
        #base_dir = "./new_dtuner/dtuner/bf-viggo-enrich8-{}/".format(ENRICH_LVL-1)   #"./new_dtuner/dtuner/bf-viggo-enrich5-{}/".format(ENRICH_LVL-1)
        base_dir = "./new_dtuner/dtuner/bf-viggo-enrich-11key-{}/".format(ENRICH_LVL-1)
#        base_dir = "./new_dtuner/dtuner/bf-viggo-answer-11key-{}/".format(ENRICH_LVL-1)

        if ENRICH_LVL == 0:
            sets = ["viggo-valid.csv", "viggo-test.csv", "viggo-train.csv"] 
        else:
            sets = ["viggo-valid.csv", "viggo-test.csv", "viggo-train.csv"]

        _nb_proc = 3
        l_proc = []
        for i in range(_nb_proc):
            l_proc.append(Process(target=startProcesses, args=(base_dir+sets[i],)))
            print('%d' % i)

        for i in range(_nb_proc):
            l_proc[i].start()
            print('start %d' % i)
        for i in range(_nb_proc):
            l_proc[i].join()
    """
    for _dir in tqdm(sets):
        if ENRICH_LVL == 0:
            p = pd.read_csv(base_dir + _dir)
        else:
            _dir = _dir.format(ENRICH_LVL - 1)
            p = pd.read_csv(base_dir + _dir)

        questions = p["ref"].tolist()
      #  answers = p["answers"].tolist()
        mrs = p["mr"].tolist()

        new_questions = ["" for i in range(len(questions))]
 #       new_answers = ["" for i in range(len(answers))]
        TEMP =  mrs[:]
        for cnt, quest in enumerate(tqdm(questions)):
            #print(cnt)
      #      ans = answers[cnt]
            TEMP = mrs[cnt]
            #print(TEMP)
            q, new_mr = enrich(graph, quest, TEMP, ENRICH_LVL)
       #     q, _ = enrich(graph, quest, TEMP, ENRICH_LVL)
       #     a, _ = enrich(graph, ans, TEMP, ENRICH_LVL)
            #print(q)
            new_questions[cnt] = q[0]
            mrs[cnt] = new_mr
       #     new_answers[cnt] = a[0]
        qa1_lst = list(zip(mrs, new_questions)) #, new_answers))
        qa1_df = pd.DataFrame(qa1_lst, columns=["mr", "ref"]) #, "answers"])
#        qa1_lst = list(zip(new_questions, new_answers))
#        qa1_df = pd.DataFrame(qa1_lst, columns=["questions", "answers"])


#        qa1_df.to_csv("new_dtuner/dtuner/bf-viggo-answer2-2/"+_dir.split("/")[-1].split(".")[0][:-2] + "_{}".format(ENRICH_LVL) + ".csv", index=False)
        if not os.path.exists("new_dtuner/dtuner/bf-viggo-enrich4-{}/".format(ENRICH_LVL)):
            os.makedirs("new_dtuner/dtuner/bf-viggo-enrich4-{}/".format(ENRICH_LVL), exist_ok=False)
        qa1_df.to_csv("new_dtuner/dtuner/bf-viggo-enrich4-{}/".format(ENRICH_LVL)+_dir.split("/")[-1].split(".")[0] + ".csv", index=False)

        '''
        if ENRICH_LVL == 0:
            trn, tst = train_test_split(qa1_lst, test_size=0.3)
            for i in [(trn, "train"), (tst, "test")]:
                format_clean(i[0], i[1])
            test_df = pd.DataFrame(tst, columns=["questions", "answers"])
            train_df = pd.DataFrame(trn, columns=["questions", "answers"])

            test_df.to_csv(_dir.split("\\")[-1].split(".")[0] + "_enrich_{}_test2.csv".format(ENRICH_LVL), index=False)
            train_df.to_csv(_dir.split("\\")[-1].split(".")[0] + "_enrich_{}_train2.csv".format(ENRICH_LVL), index=False)
        else:
            qa1_df.to_csv(_dir.split("\\")[-1].split("_{}".format(ENRICH_LVL-1))[0] + "_{}".format(ENRICH_LVL) +
                          _dir.split("\\")[-1].split("_{}".format(ENRICH_LVL-1))[1], index=False)

            format_clean(qa1_lst, _dir.split(".")[0].split("_")[-1])
    '''
    """
