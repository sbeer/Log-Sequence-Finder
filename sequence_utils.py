import string
import random
from collections import Counter
from math import erf, sqrt

import scipy as sp
import itertools
import scipy.stats as st
import numpy as np


LOG_SIZE=2000000
PATTERN_PERCENTAGE = 0.02
INJECTION_PATTERN = 'AABA'
ALPHABET_SIZE = 60 # Should be less than 62
MIN_PROB = 0.98

MAX_SEARCH_PATTERN_SIZE=10
MOSTCOMMON=3


def z2p(z):
    return 0.5 * (1 + erf(z / sqrt(2)))

import math

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

class LogGenerator(object):
    def __init__(self, 
                 name, 
                 log_size,
                 pattern_injection_percentage,
                 injection_pattern,
                 alphabet_size):
        self.name = name
        self.log_size = log_size
        self.pattern_injection_percentage = pattern_injection_percentage
        self.injection_pattern = injection_pattern
        self.alphabet_size = alphabet_size
        self.inj_pattern_ind = sorted(random.sample(range(self.log_size), int(self.pattern_injection_percentage*self.log_size*len(self.injection_pattern))))
        
    @property
    def charset_alphabet(self):
        return string.ascii_uppercase + string.ascii_lowercase + string.digits
    
    @property
    def charset_len(self):
        return len(self.charset_alphabet)
    
    @property
    def inj_pattern_data(self):
        return list(self.injection_pattern)*int(self.log_size*self.pattern_injection_percentage)
    
    @property
    def available_alphabet(self):
        assert(self.alphabet_size<=len(self.charset_alphabet))
        return ''.join(list(set(self.injection_pattern).intersection(self.charset_alphabet))
            + list(set(self.charset_alphabet) - set(self.injection_pattern).intersection(set(self.charset_alphabet)))[0:self.alphabet_size-len(self.injection_pattern)+1])
    
    @property
    def value(self):
        return ''.join(self.log)
    
    def generate(self):
        self.log = [random.choice(self.available_alphabet) for _ in range(self.log_size)]
        self.inject_pattern()
        
    def inject_pattern(self):
        for ind,data in zip(self.inj_pattern_ind,self.inj_pattern_data):
            self.log[ind] = data

			

class SearchCommonSeq(object):
    def __init__(self, 
                 log,):
        self.log = log
        self.candidates = {1:set(log.value)}
        self.frequency = {}
        
    @property
    def most_common(self):
        return MOSTCOMMON
    
    @property
    def min_prob(self):
        return MIN_PROB
    
    @property
    def max_set_size(self):
        return MAX_SEARCH_PATTERN_SIZE
    
    @lazy
    def alphabet(self):
        return set(list(self.log.value))
    
    @lazy
    def itemset(self):
        return self.generate_itemsets(self.log.value,self.max_set_size)
        
    @lazy
    def log_size(self):
        return len(self.log.value)
       
    def nsplit(self, s, n):
        return [''.join(sorted(s[k:k+n])) for k in xrange(0, len(s)-n+1, 1)]

    def generate_itemsets(self, log, M):
        itemsets = {}
        for i in range(2,M+1):
            itemsets.update({i:self.nsplit(log,i)})
        return itemsets

    def calc_frequency(self, candidates, itemset):
        freq = {}
        for i in candidates:
            freq.update({i:sum([1.0 for j in itemset if i in j])}) #/(self.log_size-len(i)+1)
        return freq
    
    def filter_high_prob(self,freq,i):
        cand, vals = zip(*freq.items())
        mu = (self.log_size-i+1) / nCr(len(self.alphabet)+i-1,i)
        sigma = np.sqrt(mu) #poisson approximation to binomial 
        z_score = [(v-mu)/sigma for v in vals]
        p = [z2p(s) for s in z_score]
        return dict([(k,v) for k,v in dict(zip(cand,p)).iteritems() if v > self.min_prob])
    
    def generate_all_candidates(self,alphabet,prev_cand):
        return set([''.join(sorted(x)) for x in itertools.product(alphabet,prev_cand)])
    
    def search_common_sequences(self):
        for i in range(2,MAX_SEARCH_PATTERN_SIZE):
            all_cand = self.generate_all_candidates(self.alphabet,self.candidates.get(i-1))
            self.frequency.update({i:self.calc_frequency(all_cand, self.itemset.get(i))})
            self.candidates.update({i:self.filter_high_prob(self.frequency.get(i),i)})
            
            
class TopCandSearchCommonSeq(SearchCommonSeq):
    def __init__(self, 
             log,**params):
        super(TopCandSearchCommonSeq,self).__init__(log,**params)
    
    def filter_high_prob(self,freq,i):
        return dict(Counter(freq).most_common(self.most_common))
        			
			
			
