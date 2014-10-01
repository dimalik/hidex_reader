"""
Helper classes to load HiDEx (Shaoul & Westbury, 2006; 2010) and S-Space
(https://github.com/fozziethebeat/S-Space) files into the Python interpreter.

Initialise a model with e.g.:
>>> model = HidexModel(PATH_TO_DICT, PATH_TO_GCM)
or
>>> model = SSpaceModel(PATH_TO_SSPACE)

PATH_TO_DICT should point to the .dict file output by HiDEx
(e.g. combined.dict) and PATH_TO_GCM to the corresponding .gcm.
In the case of S-Space, PATH_TO_SSPACE should point to the .sspace file which
has to be saved using TEXT (i.e. not binary) format.

You can perform various word tasks with the model. Some of them
are already built-in:

>>> model.get_similarity('dog', 'cat') <- cosine similarity
>>> model.get_neighbourhood('dog', topn=N) <- closest N neighbours using cosine
    similarity
>>> model.get_arc('dog') <- get neighbourhood size/density
"""

__author__ = 'Dimitrios Alikaniotis'
__email__ = 'da352@cam.ac.uk'
__affiliation__ = '\
                University of Cambridge\
                Department of Theoretical and Applied Linguistics'

import os
import sys
import re
import logging

import numpy as np
from scipy.stats.mstats import zscore

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

class GenericReader(object):
    def __init__(self, file):
        self.file = file
        
    def __iter__(self):
        with open(self.file) as corpus:
            for line in corpus:
                yield line
        
class ARCObject(dict):
    '''
    read first the comments in the get_arc method in the model class
    str(model.get_arc(word)) 
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
        
    def _init(self):
        '''
        normalises each vector to unit length
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
        
    def get_similarity(self, w1, w2):
        #TODO: implement other similarity metrics
        try:
            words = [self.model[self.vocab.token2id[i]] for i in (w1,w2,)]
        except KeyError:
            raise
        return self.get_cosine(*words)
        
    def get_neighbourhood(self, word, topn=10):
        w_ind = self.vocab.token2id[word]
        dists = np.dot(self.model, self.model[w_ind])
        best = np.argsort(dists)[::-1]
        result = [(self.vocab[ind], float(dists[ind])) for ind in best if ind != w_ind]
        return result[:topn] if topn else result ## topn=0 shows all
        
    def get_arc(self, word):
        '''
        implements the neighbourhood size/density algorithm described in Alikaniotis (2014)
        
        the algorithm simply extends the proposal of Shaoul & Westbury (2006) by relativising
        the threshold for EACH word. That is, we simply ask if the potential neighbour stands
        closer to the target word or further away than the average pairing of the target with
        any other word. In our implementation of the calculation of the semantic distances we
        take A ⋅ w_i for each word (see get_neighbourhood method with topn=0), where A is the
        ∣V∣ × D or the ∣V∣ × ∣V∣ vocabulary matrix and w_i is the target word. Having normalised
        each vector in the matrix to unit length, (see the _init method) this operation is
        equivalent to taking the cosine similarity between the word in question and all the
        other words in the vocabulary.
        The resulting ∣V∣ dimensional vector contains effectively the similarity values between
        the word in question and all the other words in the matrix. From this point it is a
        trivial task to obtain descriptive statistics for these distributions.
        Converting, thus, the vector of similarities into z-scores, we are able to keep all
        the scores above a predefined threshold obtaining thus neighbourhood size and density
        for each word, taking into account its similarity to all the other words in the lexicon.
        
        The number of stdevs above which a word is considered a neighbour, does not have to be
        explicitly set, the list_ variable reports the size and the density of the neighbourhood
        in predefined steps.
        '''
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
# hidex.get_similarity('dog', 'cat')
# ## demo topN
# hidex.get_neighbourhood('dog', topn=20)
# ## demo arc
# hidex.get_arc('dog')
# ## for better printing
# str(hidex.get_arc('dog'))
# sspace = SSpaceModel(PATH_TO_SSPACE)