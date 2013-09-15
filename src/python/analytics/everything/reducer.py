#!/usr/bin/env python

from itertools import groupby
from operator import itemgetter
import re
import sys
 
def read_mapper_output(file, separator='\t'):
    for line in file:
        yield line.rstrip().split(separator, 1)
 
def main(separator='\t'):

    USER_KEY = re.compile("^USER_")
    SESSION_KEY = re.compile("^SESSION_")

    data = read_mapper_output(sys.stdin, separator=separator)
    
    for key, group in groupby(data, itemgetter(0)):
        try:
            if re.match(USER_KEY, key):
                unique_clicks = set()
                unique_sessions = set()
                unique_shown = set()
                unique_in_cart = set()
                unique_bought = set()
                occurrences = 0
                for _, value in group:
                    occurrences += 1
                    session, clicks, shown, incart, bought = value.split("#")
                    unique_sessions.add(session)
                    unique_clicks.update(v for v in clicks.split('@') if v != '')
                    unique_shown.update(v for v in shown.split('@') if v != '')
                    unique_in_cart.update(v for v in incart.split('@') if v != '')
                    unique_bought.update(v for v in bought.split('@') if v != '')
                print "%s\t%s\t%s\t%s\t%s\t%s\t%s" % (key, occurrences, len(unique_sessions),
                    len(unique_clicks), len(unique_shown), len(unique_in_cart), len(unique_bought))
            
            elif re.match(SESSION_KEY, key):
                unique_clicks = set()
                unique_shown = set()
                unique_in_cart = set()
                unique_bought = set()
                occurrences = 0
                for _, value in group:
                    occurrences += 1
                    clicks, shown, incart, bought = value.split("#")
                    unique_clicks.update(v for v in clicks.split('@') if v != '')
                    unique_shown.update(v for v in shown.split('@') if v != '')
                    unique_in_cart.update(v for v in incart.split('@') if v != '')
                    unique_bought.update(v for v in bought.split('@') if v != '')
                print "%s\t%s\t%s\t%s\t%s\t%s" % (key, occurrences, len(unique_clicks), len(unique_shown),
                    len(unique_in_cart), len(unique_bought))

        except Exception:
            pass
 
if __name__ == "__main__":
    main()