log = LogGenerator(name='myLog',log_size=LOG_SIZE,pattern_injection_percentage=PATTERN_PERCENTAGE,injection_pattern=INJECTION_PATTERN,alphabet_size=ALPHABET_SIZE)
log.generate()

log.value[0:100]

searcher1 = TopCandSearchCommonSeq(log)
searcher1.search_common_sequences()

import numpy as np
mean = {}
all_cand = {}
for k in searcher1.candidates:
    if k>1:
        cand = {}
        mean[k] = np.mean(searcher1.candidates.get(k).values())
        for j in searcher1.candidates.get(k):
            cand[j]=searcher1.candidates.get(k)[j]/mean[k]
        all_cand[k] = cand
		
		
for k in all_cand:
        print k,all_cand.get(k)