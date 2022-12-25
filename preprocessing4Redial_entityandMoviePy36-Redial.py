import os
import json
import ast
import nltk
from nltk.tokenize import word_tokenize
import csv
import sys
import re
import pickle
import torch
import numpy as np

# from ibm_watson import NaturalLanguageUnderstandingV1
# from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
# from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions, SentimentOptions

# apikey = 'en0aZ4pMsVv6aI2m_oT8zQIsBuw-S9x0821icvzI3Sk0'
# authenticator = IAMAuthenticator(apikey)
# nlu = NaturalLanguageUnderstandingV1(
#     version='2021-08-01',
#     authenticator=authenticator
# )
# urlservice='https://api.au-syd.natural-language-understanding.watson.cloud.ibm.com/instances/85c83899-3a9a-4d0f-964a-1d8d93703bb0'
# nlu.set_service_url(urlservice)


#uncomment  entity2entityId = pickle.load(open('../data/redial/entity2entityId.pkl', 'rb')); entities.append(entity2entityId[entity]); sequence.append(entity2entityId[entity]); # count_e=50000 to # pickle.dump(e2eId, open('../data/redial/entity2entityId.pkl', 'wb')) and then run
#comment  entity2entityId = pickle.load(open('../data/redial/entity2entityId.pkl', 'rb')); entities.append(entity2entityId[entity]); sequence.append(entity2entityId[entity]); # count_e=50000 to # pickle.dump(e2eId, open('../data/redial/entity2entityId.pkl', 'wb')) and then run again


redial_path="data/redial/"
line_index = 0
all_enti = []
entity2entityId = pickle.load(open('data/redial/entity2entityId.pkl', 'rb'))
text_en_dict = pickle.load(open('data/redial/text_dict.pkl', 'rb'))

cout=0
for i in text_en_dict.items():
    if cout<=20:
        print(i)
    cout+=1


def Get_movie2id():
    with open(os.path.join(redial_path, "movies_merged.csv"), 'r', newline='', encoding='utf8') as f:
        reader = csv.reader(f)
        movie2id = {}
        for row in reader:
            if row[1] != 'movieName':
                #print row
                movie2id[row[2]]=int(row[0])+1
    return movie2id

def tokenize(message):
    """
    Text processing: Sentence tokenize, then concatenate the word_tokenize of each sentence. Then lower.
    :param message:
    :return:
    """
    sentences = nltk.sent_tokenize(message)
    tokenized = []
    for sentence in sentences:
        tokenized += nltk.word_tokenize(sentence)
    return [word.lower() for word in tokenized]

def extract_sequence(data,movie2id,fileout):
    global line_index
    for raw_line in open(os.path.join(redial_path, data)):#train_test.jsonl: combined data file from ReDial train,valid,test
        line_index +=1
        u_dict=dict()
        sequence=[]
        entities = []
        movies = []
        instance=json.loads(raw_line)
        #instance=ast.literal_eval(raw_line)
        messages=instance['messages']
        movieMentions=instance['movieMentions']
        seekerid = instance["initiatorWorkerId"]
        recommenderid = instance["respondentWorkerId"]
        seq_ind=1

        # dialogue=' '.join([message['text'] for message in messages])
        # mov1=re.findall(r'@(\d+)',dialogue)
        # mov_intext1=['@'+x for x in mov1]
        # if len(mov_intext1)>0:
        #     features=Features(sentiment=SentimentOptions(targets=mov_intext1))
        #     response=nlu.analyze(features=features, text=dialogue).get_result()
        #     sentimenttargetslist=response['sentiment']['targets']
        #     sentimenttargetsdict={x['text']:x['label'] for x in sentimenttargetslist}

        for message in messages:
            text=message['text']
            sender_id = message['senderWorkerId']

            if text =='':
                continue

            text_en = text_en_dict[text]
            mov=re.findall(r'@(\d+)',message["text"])

            for entity in text_en:
                try:
                    all_enti.append(entity)
                    entities.append(entity2entityId[entity])
                    sequence.append(entity2entityId[entity])
                    fileout.write(str(line_index) + " " + str(entity2entityId[entity])+ " " + "-1")
                    fileout.write("\n")
                except:
                    print(entity)

            for token_replace in mov:
                if sender_id == recommenderid:
                    seqstr='|'.join([str(elem) for elem in sequence])
                    # if ('@'+token_replace) in sentimenttargetsdict:
                    #     if sentimenttargetsdict['@'+token_replace] == 'negative':
                    #         continue
                    fileout.write(str(line_index) + " " + str(movie2id[token_replace])+ " " + str(seq_ind))
                    fileout.write("\n")
                    seq_ind+=1
                else:
                    # if ('@'+token_replace) in sentimenttargetsdict:
                    #     if sentimenttargetsdict['@'+token_replace] == 'negative':#'positive' 'neutral' 'negative'
                    #         continue
                    fileout.write(str(line_index) + " " + str(movie2id[token_replace])+ " " + "0")
                    fileout.write("\n")
                movies.append(movie2id[token_replace])
                sequence.append(movie2id[token_replace])


            # tokens = word_tokenize(text)
            # for i in range(len(tokens)):
            #     if '@' in tokens[i] and (i + 1) != len(tokens):
            #         #sequence.append(tokens[i+1])
            #         if tokens[i+1] not in movie2id.keys():
            #             if '.' in tokens[i+1]: # deal with tokenization error
            #                 token_replace=tokens[i + 1][0:tokens[i+1].index('.')]
            #                 sequence.append(movie2id[token_replace])
            #                 fileout.write(str(line_index) + " " + str(movie2id[token_replace]))
            #                 fileout.write("\n")
            #             # else:
            #             #     count_namematch = 0
            #             #     for movieMention in movieMentions.items():
            #             #         if tokens[i + 1].lower() == movieMention[1].split(' ')[0].lower():
            #             #             count_namematch+=1
            #             #             tokenname2id=movieMention[0]
            #             #     if count_namematch==1:
            #             #         print "++++++++++++++++++"
            #             #         print str(line_index) + " " + str(movie2id[tokenname2id])
            #             #         print text
            #             #         print "++++++++++++++++++"
            #             #         sequence.append(movie2id[tokenname2id])
            #             #         fileout.write(str(line_index) + " " + str(movie2id[tokenname2id]))
            #             #         fileout.write("\n")
            #             #     else:
            #             #         print line_index
            #             #         print "movie not found:"+ tokens[i+1]
            #             #         print text
            #         else:
            #             sequence.append(movie2id[tokens[i+1]])
            #             fileout.write(str(line_index)+" "+str(movie2id[tokens[i+1]]))
            #             fileout.write("\n")
        u_dict[line_index]=sequence
        if len(sequence)<=1:
            print("**********************")
            print(line_index)
            print(sequence)


def main():

    movie2id=Get_movie2id()

    fileout = open(os.path.join(redial_path, "redial_process_test.txt"), 'w+')
    extract_sequence("train_data", movie2id, fileout)
    print("last train index"+str(line_index))
    # fileout.flush()
    # fileout.close()
    # fileout = open(os.path.join(redial_path, "test_process.txt"), 'w+')
    extract_sequence("valid_data", movie2id, fileout)
    print("last valid index"+str(line_index))
    extract_sequence("test_data", movie2id, fileout)
    fileout.flush()
    fileout.close()

    # count_e=50000
    # e2eId={}
    # for e in all_enti:
    #     if e not in e2eId:
    #         count_e+=1
    #         e2eId[e] = count_e
    # pickle.dump(e2eId, open('../data/redial/entity2entityId.pkl', 'wb'))
    #

if __name__ == "__main__":
    main()


