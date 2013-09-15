#!/usr/bin/env python

from itertools import groupby
from operator import itemgetter
from collections import defaultdict
import sys

def read_mapper_output(file, separator='\t'):
    for line in file:
        yield line.rstrip().split(separator, 1)
 
def main(separator='\t'):

    THRESHOLD = 50

    data = read_mapper_output(sys.stdin, separator=separator)
    
    for user, group in groupby(data, itemgetter(0)):
        try:
            if user == '':
                continue
            unique_clicks = set()
            key_to_label = defaultdict(lambda: 0)
            for _, value in group:

                _, clicks, shown, incart, bought = value.strip().split("#")

                for label, items_group in enumerate((shown, clicks, incart, bought), start=1):

                    if items_group is clicks:
                        is_clicks = True
                    else:
                        is_clicks = False

                    for item in items_group.split('@'):
                        if item == '':
                            continue
                        key_to_label[item] = label      # an update is always advantageous
                        if is_clicks:                   # corresponds to clicked list.
                            unique_clicks.add(item)

            if len(unique_clicks) >= THRESHOLD:
                for key in key_to_label:
                    print "%s\t%s\t%s" % (user, key, key_to_label[key])

        except Exception:
            pass
 
if __name__ == "__main__":
    main()