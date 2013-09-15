#!/usr/bin/env python

import sys
import json

for line in sys.stdin:
  try:
    line = json.loads(line)
    incart = set(x['ItemId'] for x in line['clicks'] if x['InCart'] == 'true')
    bought = set(x['ItemId'] for x in line['clicks'] if x['Ordered'] == 'true')
    print '%s%s%s' % ('USER_' + line['visitorid'], "\t", 
        line['wmsessionid'] + '#' + '@'.join(str(x) for x in line['clickeditems']) + '#' + '@'.join(str(x) for x in line['shownitems']) +
        '#' + '@'.join(incart) + '#' + '@'.join(bought))
    print '%s%s%s' % ('SESSION_' + line['wmsessionid'], "\t",
        '@'.join(str(x) for x in line['clickeditems']) + '#' + '@'.join(str(x) for x in line['shownitems']) + '#' +
        '@'.join(incart) + '#' + '@'.join(bought))
  except Exception:
    pass
