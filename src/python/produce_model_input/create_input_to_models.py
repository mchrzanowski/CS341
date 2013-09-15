from collections import defaultdict
import random
from math import ceil, floor

def create(filename, training_file, testing_file):

    RATIO = 0.8
    
    users_to_items = defaultdict(list)
    with open(filename, 'rb') as f:
        for line in f:
            user, item, rating = line.split()
            users_to_items[user].append((item, rating))


    for user in users_to_items:
        random.shuffle(users_to_items[user])

    training = open(training_file, 'wb')
    testing = open(testing_file, 'wb')


    for user in users_to_items:
        split_point = int(floor(RATIO * len(users_to_items[user])))
        for (item, rating) in users_to_items[user][:split_point]:
            training.write('%s\t%s\t%s\n' % (user, item, rating))
        for (item, rating) in users_to_items[user][split_point:]:   
            testing.write('%s\t%s\t%s\n' % (user, item, rating))

    training.close()
    testing.close()


if __name__ == "__main__":
    import argparse
    import time
    start_time = time.time()
    parser = parser = argparse.ArgumentParser(description="Create Model Training and Testing File Input.")
    parser.add_argument('-input', default='./good_input', type=str, help="Input file.")
    parser.add_argument('-train', default='./good_training', type=str, help="Training file to produce.")
    parser.add_argument('-test', default='./good_testing', type=str, help="Testing file to produce.")

    args = vars(parser.parse_args())
    create(args['input'], args['train'], args['test'])
    end_time = time.time()
    print "Runtime: {} seconds.".format(end_time - start_time)
