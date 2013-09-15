import json
import re

def convert(line):

    titles = re.findall("title\":\".*?\"", line)

    if len(titles) < 1:
        return None, None

    title = titles[0]

    title = re.sub("\"", "", title)
    title = re.sub("title:", "", title)

    id = re.findall("id\":\".*?\"", line)

    assert len(id) == 1
    id = id[0]

    id = re.sub("\"", "", id)
    id = re.sub("id:", "", id)

    return title, id

def main(data, output):

    items_to_categories = dict()
    output_file = open(output, 'wb')
    with open(data, 'rb') as f:
        for line in f:
            title, id = convert(line)

            if title is None or id is None:
                continue

            id = int(id)

            output_file.write('%s\t%s\n' % (id, title))


if __name__ == "__main__":
    import argparse
    import time
    start_time = time.time()
    parser = parser = argparse.ArgumentParser(description="Create User-Category Files from User-Item & Walmart Reference Files.")
    parser.add_argument('-data', type=str, help="Walmart Item Attribute Data.")
    parser.add_argument('-output', default='./item_to_category', type=str, help="File to produce.")

    args = vars(parser.parse_args())
    main(args['data'], args['output'])
    end_time = time.time()
    print "Runtime: {} seconds.".format(end_time - start_time)

            