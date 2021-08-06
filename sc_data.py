
import os
import numpy as np
import pandas as pd

from os import listdir
from os.path import isfile, join
import re
import csv




class InputData(object):

    def __init__(self,bulk = "" , singleCell="" , result="", isVAF = 1):

        self.isVAF = isVAF
        self.singleCell_path = singleCell
        self.bulk_path = bulk
        self.resultPath = "./output"
        self.singelCell_Mutations_As_Documnet_path = "./input/mutations.csv"
        self.cells = []
        self.vocab = []


    def fetchData(self):

        df = pd.read_csv(self.singleCell_path)
        counter = 0

        for i in range(0,df.shape[0]):
            k=0
            for j in range(0,df.shape[1]):
                if df.iat[i,j]==1:
                    k=k+1
                    if counter < k :
                        counter = k

        header = []
        for j in range(0,counter):
            header.append(j+1)


        empty_row = np.zeros(df.shape[0], dtype=np.object)
        for i in range(0,df.shape[0]):
            empty_row[i] = 1
            cnt = 0
            for j in range(0,df.shape[1]):
                if df.iat[i,j]==1:
                    cnt = cnt+1
            if cnt > 0 :
             empty_row[i] = 0

        row_number = -1
        rows = []
        for i in range(0,df.shape[0]):
            if empty_row[i]==0:
                row_number+=1
                row =[]
                k=0
                for j in range(0,df.shape[1]):
                    if df.iat[i,j]==1:
                        row.append('X'+str(j+1))
                        k=k+1
                rows.append(row)

        
        with open(self.singelCell_Mutations_As_Documnet_path, 'w+', newline='') as file:
         writer = csv.writer(file)
         writer.writerow(header)
         writer.writerows(rows)


        vocab_mat = np.zeros((len(rows), len(header)), dtype=np.object)
        word_count = 0

        VAF = []
        with open(self.bulk_path) as csv_file:   
          csv_reader = csv.reader(csv_file, delimiter=',')
          line_count = -1
          for row in csv_reader:
             line_count = line_count+1
             if line_count != 0:
               rr = [float(i) for i in ' '.join(row).split()]
               VAF.append(rr)

    
        doc = []
        corpus = []
        for i in range(len(rows)):
            for j in range(len(rows[i])):
                if (rows[i])[j] == (rows[i])[j]:
                    word = (rows[i])[j]
                    vocab_mat[i, j] = int(((rows[i])[j])[1:])
                    doc.append(word)
                    word_count += 1
            corpus.append(doc)
            doc = []
        self.vocab = vocab_mat.flatten().tolist()
        self.vocab = list(filter(lambda a: a!=0,self.vocab))

        self.vocab = list(dict.fromkeys(self.vocab))
        self.vocab = sorted(list(self.vocab))

        for i in range(len(self.vocab)):
            self.vocab[i] = 'X'+str(self.vocab[i])

            
        new_corpus = []
        for doc in corpus:
            new_doc = []
            for word in doc:
                word_idx = int(word[1:]) - 1
                new_doc.append(word_idx)
            new_corpus.append(new_doc)        
            
        
        all_word_features = []
        for a in range(len(VAF[0])): 
           f = []
           for v in range(len(VAF)):
               f.append(VAF[v][a]) 
           all_word_features.append(f)
        
        features = [] 
        for doc in corpus:
            doc_feature = []
            for word in doc:
                word_idx = int(word[1:]) - 1
                doc_feature.append(all_word_features[word_idx])
            features.append(doc_feature)    
        
        features = np.array(list(features))
        all_word_features = np.array(list(all_word_features))
        
        return new_corpus , VAF , all_word_features, features 