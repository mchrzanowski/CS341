from collections import defaultdict


def generate_stats(keys_to_freq):
    '''
        calculate mean, median, st dev. 
        input is a dict of key -> frequency of keys. keys and values are numbers.
    '''
    total_items = sum(keys_to_freq[x] for x in keys_to_freq)
    mean = sum(keys_to_freq[x] * x for x in keys_to_freq) / float(total_items)
    
    stdev = 0
    for number in keys_to_freq:
        stdev += keys_to_freq[number] * ((number - mean) ** 2) 
    stdev /= float(total_items)
    stdev = stdev ** 0.5

    half = total_items / 2.
    median = None
    first_half = None
    for number in sorted(keys_to_freq):
        half -= keys_to_freq[number]
        if half <= 0:
            if first_half == None:
                median = number
            else:
                median = (number + first_half) / 2.
        elif half < 1:
            first_half = number
        if median is not None:
            break

    return mean, median, stdev

def return_formatted_statistics(keys_to_freq):
    mean, median, stdev = generate_stats(keys_to_freq)
    return "Mean: {}\nMedian: {}\nSt Dev: {}\n".format(mean, median, stdev)

def produce_session_stats(file):
    occurrences_freq = defaultdict(lambda: 0)
    number_of_sessions_freq = defaultdict(lambda: 0)
    number_of_shown_freq = defaultdict(lambda: 0)
    number_of_clicked_freq = defaultdict(lambda: 0)
    number_of_incart_freq = defaultdict(lambda: 0)
    number_of_bought_freq = defaultdict(lambda: 0)

    with open(file, 'rb') as f:
        for line in f:
            user, occurrences, clicked, shown, incart, ordered = line.strip().split('\t')
            occurrences = int(occurrences)
            shown = int(shown)
            clicked = int(clicked)
            incart = int(incart)
            ordered = int(ordered)

            occurrences_freq[occurrences] += 1
            number_of_shown_freq[shown] += 1
            number_of_clicked_freq[clicked] += 1
            number_of_incart_freq[incart] += 1
            number_of_bought_freq[ordered] += 1

    # now, generate stats.
    print "Occurrences of Session:"
    print return_formatted_statistics(occurrences_freq)

    print "Unique Items Shown per Session:"
    print return_formatted_statistics(number_of_shown_freq)

    print "Unique Clicks per Session:"
    print return_formatted_statistics(number_of_clicked_freq)

    print "Unique Items Placed In Cart per Session:"
    print return_formatted_statistics(number_of_incart_freq)

    print "Unique Items Ordered per Session:"
    print return_formatted_statistics(number_of_bought_freq)


def produce_user_stats(file):
    occurrences_freq = defaultdict(lambda: 0)
    number_of_sessions_freq = defaultdict(lambda: 0)
    number_of_shown_freq = defaultdict(lambda: 0)
    number_of_clicked_freq = defaultdict(lambda: 0)
    number_of_incart_freq = defaultdict(lambda: 0)
    number_of_bought_freq = defaultdict(lambda: 0)

    with open(file, 'rb') as f:
        for line in f:
            user, occurrences, sessions, clicked, shown, incart, ordered = line.strip().split('\t')
            occurrences = int(occurrences)
            sessions = int(sessions)
            shown = int(shown)
            clicked = int(clicked)
            incart = int(incart)
            ordered = int(ordered)

            occurrences_freq[occurrences] += 1
            number_of_sessions_freq[sessions] += 1
            number_of_shown_freq[shown] += 1
            number_of_clicked_freq[clicked] += 1
            number_of_incart_freq[incart] += 1
            number_of_bought_freq[ordered] += 1

    # now, generate stats.
    print "Occurrences of User:"
    print return_formatted_statistics(occurrences_freq)

    print "Sessions per User:"
    print return_formatted_statistics(number_of_sessions_freq)

    print "Unique Items Shown per User:"
    print return_formatted_statistics(number_of_shown_freq)

    print "Unique Clicks per User:"
    print return_formatted_statistics(number_of_clicked_freq)

    print "Unique Items Placed In Cart Per User:"
    print return_formatted_statistics(number_of_incart_freq)

    print "Unique Items Ordered Per User:"
    print return_formatted_statistics(number_of_bought_freq)


def main(part, file):
    if part.upper() == 'U':
        produce_user_stats(file)
    elif part.upper() == "S":
        produce_session_stats(file)

if __name__ == "__main__":
    import argparse
    import time

    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-part', required=True, type=str, help="File contains U[ser] or S[ession] data")
    parser.add_argument('-file', required=True, type=str, help="File path.")
    args = vars(parser.parse_args())
    main(args['part'], args['file'])

    end = time.time()
    print "Runtime: {} seconds.".format(end - start)
