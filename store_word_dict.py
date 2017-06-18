import numpy as np
from numpy.linalg import norm
import cPickle

def storeWordVecsDict(wordVecsFileName, dictFileName):
    """Store the word vectors given in a text file into a dict
    (after normalizing the vectors), and save the dict as a pickle
    file. The vectors are numpy float arrays.
    """
    with open(wordVecsFileName, 'r') as wordVecsFile:
        lineNum = 1
        wordVecsDict = {}
        for line in wordVecsFile:
            # Skip over line 1 in the word2vec file
            if lineNum != 1 or wordVecsFileName.startswith('glove'):
                parts = line.split()
                word = parts[0]
                values = parts[1:]
                vector = np.array([float(v) for v in values])
                wordVecsDict[word] = vector / norm(vector)

                if lineNum % 100 == 0:
                    print 'Line', lineNum, 'processed'

            lineNum += 1
                    
        with open(dictFileName, 'wb') as dictFile:
            cPickle.dump(wordVecsDict, dictFile, cPickle.HIGHEST_PROTOCOL)

            
import os
if not os.path.isdir('dicts'):
    os.makedirs('dicts')

# Can choose either word2vec or GloVe vectors
wordVecsFileName = 'GoogleNews-vectors-negative300.txt'#'glove/crawl/glove.840B.300d.txt'#
dictFileName = 'dicts/word2vec-GoogleNews-vecs300-norm.pickle'#'dicts/glove-crawl840B-vecs300-norm.pickle'#
storeWordVecsDict(wordVecsFileName, dictFileName)
