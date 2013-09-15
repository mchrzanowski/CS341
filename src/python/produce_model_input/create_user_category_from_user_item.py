import json
import re

def convert(line):

    categories = re.findall("primary_category_path\":\".*?\"", line)

    if len(categories) < 1:
        return None, None

    categories = categories[0]

    categories = re.sub("\"", "", categories)
    categories = re.sub("primary_category_path:", "", categories)

    id = re.findall("id\":\".*?\"", line)

    assert len(id) == 1
    id = id[0]

    id = re.sub("\"", "", id)
    id = re.sub("id:", "", id)

    return categories, id

def main(data, input, output):

    items_to_categories = dict()
    with open(data, 'rb') as f:
        for line in f:
            # escape everything

            categories, id = convert(line)

            if categories is None or id is None:
                continue

            categories = categories.split('a')[1:]
            id = int(id)

            items_to_categories[id] = categories

    output_file = open(output, 'wb')
    skipped = set()
    with open(input, 'rb') as f:
        for line in f:
            user, item, rating, _ = line.split(None, 3)
            item = int(item)
            if item not in items_to_categories:
                if item not in skipped:
                    skipped.add(item)
                    print "Skipped: {}".format(item)
                continue
            for category in items_to_categories[item]:
                output_file.write('%s\t%s\t%s\n' % (user, category, rating))

    output_file.close()
    print "Skipped: %s items" % len(skipped)

if __name__ == "__main__":
    import argparse
    import time
    start_time = time.time()
    parser = parser = argparse.ArgumentParser(description="Create User-Category Files from User-Item & Walmart Reference Files.")
    parser.add_argument('-data', type=str, help="Walmart Item Attribute Data.")
    parser.add_argument('-input', default='./good_training', type=str, help="Training file to User.")
    parser.add_argument('-output', default='./user_category', type=str, help="File to produce.")

    args = vars(parser.parse_args())
    main(args['data'], args['input'], args['output'])
    end_time = time.time()
    print "Runtime: {} seconds.".format(end_time - start_time)
