#!/usr/bin/env python

import sys
import json
from mrjob.job import MRJob

class MRWordFreqCount(MRJob):

    def mapper(self, _, line):
        try:
            line = line.strip()
            line = json.loads(line)
            yield (line['wmsessionid'], 1)
        except Exception:
            pass

    def combiner(self, session, counts):
        yield (session, sum(counts))

    def reducer(self, session, counts):
        yield (session, sum(counts))

if __name__ == '__main__':
    MRWordFreqCount.run()
