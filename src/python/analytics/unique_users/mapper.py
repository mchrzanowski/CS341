#!/usr/bin/env python

import sys
import json

for line in sys.stdin:
  try:
    line = line.strip()
    line = json.loads(line)
    print '%s%s%s' % ('USER_' + line['visitorid'], "\t", line['wmsessionid'] + '#' + str(len(line['clickeditems'])) + '#' + str(len(line['shownitems'])))
    print '%s%s%s' % ('SESSION_' + line['wmsessionid'], "\t", str(len(line['clickeditems'])) + '#' + str(len(line['shownitems'])))
  except Exception as e:
    pass
