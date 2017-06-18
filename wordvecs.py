import cPickle
from scipy.spatial.distance import cosine, euclidean
from operator import itemgetter
import heapq
from numpy.linalg import norm
import numpy as np

class WordVecsDict:
    """Class for dealing with word vectors and evaluating relational similarities and completing analogies using them."""

    def __init__(self):
        self.wordVecsDict = {}
        self.allWords = []
        self.allVecs = []

    def loadDict(self, dictFileName):
        """Load the word-vector dictionary from the given pickle file."""
        with open(dictFileName, 'rb') as dictFile:
            self.wordVecsDict = cPickle.load(dictFile)

        self.allWords = np.array(self.wordVecsDict.keys())
        self.allVecs = np.array(self.wordVecsDict.values()).T

    def hasWords(self, *words):
        """Determine whether all given words are in the dictionary."""
        for word in words:
            if word not in self.wordVecsDict:
                print word, 'is not in the dictionary'
                return False
        return True

    def getWordVec(self, word):
        return self.wordVecsDict[word]

    def getRelationVec(self, pair):
        """Given a pair of words, return the vector representing the relation between them."""
        vec1 = self.wordVecsDict[pair[0]]
        vec2 = self.wordVecsDict[pair[1]]
        return vec1 - vec2

    def relationalSim(self, pair1, pair2, method='cosine'):
        """Calculate the relational similarity between two pairs of words."""
        relVec1 = self.getRelationVec(pair1)
        relVec2 = self.getRelationVec(pair2)
        if method == 'cosine':
            similarity = 1 - cosine(relVec1, relVec2)
        elif method == 'euclidean':
            similarity = 1 - euclidean(relVec1, relVec2)
        return similarity
    
    def getDvec(self, wordA, wordB, wordC):
        """Given A:B::C:?, return the vector representing word D."""
        vecA = self.wordVecsDict[wordA]
        vecB = self.wordVecsDict[wordB]
        vecC = self.wordVecsDict[wordC]
        vecD = vecB - vecA + vecC
        return vecD
    
    def getClosestWords(self, vec, num_results=50, similarity='cosine', remove_words=[]):
        if similarity == 'cosine':
            vec_norm = vec / norm(vec)
            allSims = np.dot(vec_norm, self.allVecs)
        elif similarity == 'euclidean':
            allSims = 1 - norm(vec - self.allVecs.T, axis = 1)

        # Remove remove_words from the list of results
        masks = [self.allWords != word for word in remove_words]
        init_mask = np.ones(len(allSims), dtype='bool')
        mask = reduce(np.logical_and, masks, init_mask)

        # Attach each similarity to the correct word
        wordSims = zip(self.allWords[mask], allSims[mask])

        # Get the words with the highest similarities
        bestWordSims = heapq.nlargest(num_results, wordSims, key=itemgetter(1))

        return bestWordSims

    def getAnalogyCompletions(self, wordA, wordB, wordC, num_results=50, method='cosine', w=None):
        """Given A:B::C:?, return the top num_results completions for the D term."""

        # Get an imagined D vector and compute its distances to all words in the dictionary
        vecD = self.getDvec(wordA, wordB, wordC)

        if method == 'cosine':
            # Calculate cosine similarities between vector D and all word vectors
            vecD_norm = vecD / norm(vecD)
            allSims = np.dot(vecD_norm, self.allVecs)
        else:
            allSims = 1 - norm(vecD - self.allVecs.T, axis = 1)

        # Remove words A, B, and C from the list of results
        mask = np.logical_and(self.allWords != wordA, np.logical_and(self.allWords != wordB,
            self.allWords != wordC))

        # Attach each similarity to the correct word
        wordSims = zip(self.allWords[mask], allSims[mask])

        bestWordSims = heapq.nlargest(num_results, wordSims, key=itemgetter(1))

        return bestWordSims
