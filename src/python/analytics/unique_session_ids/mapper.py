#!/usr/bin/env python

import sys
import json

for line in sys.stdin:
  try:
    line = line.strip()
    line = json.loads(line)
    print '%s%s%d' % (line['wmsessionid'], "\t", 1)
  except Exception:
    pass
