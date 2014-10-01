import os
import sys
import re
import logging
import numpy as np
from six import string_types

from gensim import corpora
from scipy.stats.mstats import zscore


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

__author__ = 'Dimitrios Alikaniotis'
__email__ = 'da352@cam.ac.uk'
__affiliation__ = '\
                University of Cambridge\
                Department of Theoretical and Applied Linguistics'
                
        
class ARCObject(dict):
    '''
    customised dictionary object to hold also the name of the
    word and have a nicer printing capability
    '''
    def __init__(self, name, range_, *args, **kwargs):
        self.name = name
        self.range_ = range_
        self.update(*args, **kwargs)
        
    def __setitem__(self, key, value):
        super(ARCObject, self).__setitem__(key, value)
        
    def __str__(self):
        rng = range(2 * len(self.range_))
        template = '\t'.join(["{{{0}}}".format(x) for x in rng]) ## template form!
        keys = sorted(self.keys())
        list_ = [i for sl in [list(self[k]) for k in keys] for i in sl] ## flatten lists
        ans = []
        ## align the negatives
        i = 0
        while keys[0] != self.range_[i]:
            ans.extend([0,0])
            i += 1
        ans.extend(list_)
        ## add the missing from the top
        while len(ans) != len(rng):
            ans.extend([0,0])
        return template.format(*ans)
        
class Dictionary(dict):
    '''
    the dictionary class has two attributes
    a token2id (which stores the data) and a 
    corresponding id2token
    '''
    def __init__(self):
        self.token2id = {}
        self.id2token = {}
    
    def __getitem__(self, tokenid):
        if len(self.id2token) != len(self.token2id):
            self.id2token = {v: k for k, v in self.token2id.iteritems()}
        return self.id2token[tokenid]


class Model(object):
    '''
    all the deriving models define just different ways
    to load a vocabulary and a dictionary
    needs a dictionary defining a token2id attribute
    and a corresponding id2token
    the __getitem__ method returns id2token
    '''
    
    def _init(self):
        '''
        normalise the matrix vector by vector (this can be quite expensive
        for large matrices i.e HAL so search for alternative)
        '''
        for i in range(self.model.shape[0]):
            if np.count_nonzero(self.model[i, :]) == 0:
                continue
            else:
                self.model[i, :] /= np.sqrt((self.model[i, :] ** 2).sum(-1))
                
    def __getitem__(self, word):
        return self.model[self.vocab.token2id[word]]
    
    def get_cosine(self, w1, w2):
        return np.dot(w1, w2)
        
    def get_similarities(self, w1, w2):
        try:
            words = [self.model[self.vocab.token2id[i]] for i in (w1,w2,)]
        except KeyError:
            raise
        return self.get_cosine(*words)
        
    def get_neighbourhood(self, word, topn=10):
        '''
        returns a list of tuples containing the second word
        and their cosine similarity
        '''
        w_ind = self.vocab.token2id[word]
        logging.info('Calculating the dot product of the V x D matrix and the word vector')
        dists = np.dot(self.model, self.model[w_ind])
        best = np.argsort(dists)[::-1]
        result = [(self.vocab[ind], float(dists[ind])) for ind in best if ind != w_ind]
        return result[:topn] if topn else result
        
    def get_arc(self, word):
        list_ = np.arange(-5, 10, 1) ## range and step of reporting
        ans_list = ARCObject(word, list_)
        most_similar = self.get_neighbourhood(word, topn=0) ## get all
        zscores = zscore([sim for (w, sim) in most_similar])
        self.nl = zip(most_similar, zscores)
        ans_ncount = ans_arc = 0
        it = iter(self.nl)
        high_ind = -1
        high_val = list_[high_ind]
        try:
            low_ind = -2
            low_val = list_[low_ind]
        except IndexError:
            print "Please use a list that contains more than two items"
            raise
        for i, (k, v) in enumerate(it):
            if v < list_[0]:
                return ans_list
            while v < high_val and v < low_val:
                ans_list[low_val] = tuple([ans_ncount, ans_arc / i if ans_ncount != 0 else 0])
                ans_ncount = 0
                low_ind -= 1
                high_ind -= 1
                low_val, high_val = list_[low_ind], list_[high_ind]
            ans_ncount += 1
            ans_arc += k[1]
        return ans_list
        
class GenericReader(object):
    def __init__(self, file):
        self.file = file
        
    def __iter__(self):
        with open(self.file) as corpus:
            for line in corpus:
                yield line
            
class HidexModel(Model):
    def __init__(self, dict_file, gcm_file):
        self.dict_file = dict_file
        self.gcm_file = gcm_file
        
        self.load_dict()
        self.load_model()
    
    def load_dict(self):
        print 'Loading dictionary...',
        self.vocab = Dictionary()
        with open(self.dict_file) as dict_:
            for line in dict_:
                try:
                    w, i, freq = line.split('\t')
                    self.vocab.token2id[w] = int(i)
                except:
                    break

    def load_model(self):
        '''
        takes the gcm output and makes it a vocabulary
        '''
        gcm = GenericReader(self.gcm_file)
        logging.info('Loading model...this could take a while')
        for i, vector in enumerate(gcm, -1):
            if i == -1:
                '''
                get header info
                '''
                words, dims = vector.split()
                self.model = np.empty((int(words), int(dims)), dtype=np.float32)
            else:
                print "\rLoading... {0:.2f}%   ".format(i / float(words) * 100),
                sys.stdout.flush()
                self.model[i] = np.fromstring(vector, sep=' ')
        self._init()	
        
class SSpaceModel(Model):
    def __init__(self, sspace_file):
        self.sspace_file = sspace_file
        self.load_model()
    
    def load_model(self):
        self.vocab = Dictionary()
        sspace = GenericReader(self.sspace_file)
        logging.info('Loading model...this could take a while')
        for i, line in enumerate(sspace, -1):
            if i == -1:
                '''
                get header info
                '''
                fline = re.sub(r'(\x00[0]+)', ' ', line).split() ## dirty
                words = fline[-2]
                dims = fline[-1]
                self.model = np.empty((int(words), int(dims)), dtype=np.float32)
            else:
                print "\rLoading... {0:.2f}%   ".format(i / float(words) * 100),
                sys.stdout.flush()
                word, vector = line.split('|')
                self.vocab.token2id[word] = i
                self.model[i] = np.fromstring(vector, sep=' ')
        self._init()
        
# PATH_TO_DICT = ''
# PATH_TO_GCM = ''
# PATH_TO_SSPACE = ''
# hidex = HidexModel(PATH_TO_DICT, PATH_TO_GCM)
# ## demo similarity
# hidex.get_similarities('dog', 'cat')
# ## demo topN
# hidex.get_neighbourhood('dog', topn=20)
# ## demo arc
# hidex.get_arc('dog')
# ## for better printing
# str(hidex.get_arc('dog'))
# sspace = SSpaceModel(PATH_TO_SSPACE)