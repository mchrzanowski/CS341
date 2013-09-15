#!/usr/bin/env python

import os
import sys
import json
import time

def main():

    clicks = [0] * 60

    root = "/mnt/NAS/code/cs341/flat/"

    for folder in os.listdir(root):
        subdir = os.path.join(root, folder)
        for file in os.listdir(subdir):
            subfile = os.path.join(subdir, file)
            print "Processing: {}".format(subfile)

            with open(subfile, 'r') as f:
                for line in f:
                    try:
                        line = json.loads(line)
                        if line['visitorid'] == '':
                            continue
                        shown = line['shownitems']
                        clicked = set(x for x in line['clickeditems'])

                        for i, item in enumerate(shown):
                            if item in clicked:
                                clicks[i] += 1
   
                    except Exception:
                        pass

    print "Unnormalized..."
    for i in xrange(len(clicks)):
        print "{}\t:{}".format(i + 1, clicks[i]) 

    print "Normalized..."
    normalization = float(sum(clicks))
    for i in xrange(len(clicks)):
        clicks[i] /= normalization
        print "{}\t:{}".format(i + 1, clicks[i])


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print "Runtime: {} seconds".format(end_time - start_time)
