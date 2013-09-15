import numpy
import time

from collections import defaultdict
from math import ceil
from multiprocessing.pool import ThreadPool

def run(training_file, hidden_units, verbose=False, batch_size=100):

  user_to_items, item_translation, labels, user_translation = parse_input_file(training_file)

  ITEMS = len(item_translation)
  FEATURES = hidden_units
  SOFTMAX = len(labels)

  hidden_biases  = numpy.zeros(FEATURES)
  visible_biases = populate_with_input_data(user_to_items, ITEMS, SOFTMAX)
  weights = 0.02 * numpy.random.randn(ITEMS, SOFTMAX, FEATURES) - 0.01

  if verbose:
    print "BATCH SIZE:\t{}".format(batch_size)
    print "FEATURES:\t{}".format(FEATURES)
    print "LABELS:\t\t{}".format(SOFTMAX)
    print "ITEMS:\t\t{}".format(ITEMS)

  train(FEATURES, hidden_biases, ITEMS, SOFTMAX, user_to_items, visible_biases, weights)

  return hidden_biases, visible_biases, weights, item_translation, ITEMS, labels, user_translation, user_to_items


def populate_with_input_data(user_to_items, ITEMS, SOFTMAX):

    item_rating_count = numpy.zeros((ITEMS, SOFTMAX))

    for user in user_to_items:
        for (item, rating) in user_to_items[user]:
            item_rating_count[item, rating] += 1

    non_zero_item_rating_count = item_rating_count.copy()
    non_zero_item_rating_count += (non_zero_item_rating_count == 0)

    normalization = item_rating_count.sum(axis=1)
    normalization += (normalization == 0)

    visible_biases = numpy.divide(non_zero_item_rating_count.T, normalization).T
    visible_biases = numpy.log(visible_biases)

    activations = (item_rating_count > 0)

    visible_biases = visible_biases * activations

    return visible_biases


def parse_input_file(file, exclude_unseen=False, item_translation=None, labels=None, user_translation=None):

  user_to_items = defaultdict(set)

  if item_translation is None or labels is None or user_translation is None:
    initially_None = True
    labels = set()
    item_translation = dict()
    user_translation = dict()
  else:
    initially_None = False

  with open(file, 'rb') as f:
    for line in f:
      user, item, rating = line.split()

      if rating not in labels and not exclude_unseen:
        labels.add(rating)

      if item not in item_translation and not exclude_unseen:
        item_translation[item] = len(item_translation)

      if user not in user_translation and not exclude_unseen:
        user_translation[user] = len(user_translation)

      if item in item_translation and rating in labels and user in user_translation:
        user_to_items[user_translation[user]].add((item_translation[item], int(rating) - 1))

  if initially_None:
    return user_to_items, item_translation, labels, user_translation
  else:
    return user_to_items


def train_user(cd_steps, FEATURES, hidden_biases, ITEMS, SOFTMAX, user, user_to_items, visible_biases, weights):

    neg_hidden_probs = numpy.zeros((ITEMS, SOFTMAX))
    nvp2 = numpy.zeros((ITEMS, SOFTMAX))
    negative_visual_softmax = numpy.zeros(ITEMS)

    nrmse = 0
    ntrain = 0

    hidden_units = numpy.zeros(FEATURES)

    for (item, rating) in user_to_items[user]:
        hidden_units += weights[item, rating]

    pos_hidden_probs = sigmoid(hidden_units + hidden_biases)
    positive_hidden_states = pos_hidden_probs > numpy.random.random(pos_hidden_probs.shape)
    current_positive_hidden_states = positive_hidden_states

    # first step : error calculation.
    for (item, rating) in user_to_items[user]:
        nvp2[item] += numpy.dot(weights[item], pos_hidden_probs.T)
        nvp2[item] = sigmoid(nvp2[item] + visible_biases[item])

        normalization = sum(nvp2[item])
        nvp2[item] /= normalization

        expectedV = sum(r * nvp2[item, r] for r in xrange(SOFTMAX))
        vdelta = rating - expectedV
        nrmse += vdelta ** 2                                                    # EMIT THIS.

    ntrain += len(user_to_items[user])                                          # EMIT THIS.

    step = 0
    while step < cd_steps:
        step += 1
        final_step = step >= cd_steps

        # POSITIVE CD calculation
        for (item, _) in user_to_items[user]:
            neg_hidden_probs[item] += numpy.dot(weights[item], current_positive_hidden_states.T)
            neg_hidden_probs[item] = sigmoid(neg_hidden_probs[item] + visible_biases[item])

            normalization = sum(neg_hidden_probs[item])
            neg_hidden_probs[item] /= normalization

            rand_value = numpy.random.rand()
            for r in xrange(SOFTMAX):
                rand_value -= neg_hidden_probs[item, r]
                if rand_value <= 0:
                    negative_visual_softmax[item] = r
                    break
            if rand_value > 0:
                negative_visual_softmax[item] = SOFTMAX - 1

        # NEGATIVE CD calculation.
        hidden_units.fill(0)
        for (item, _) in user_to_items[user]:
            hidden_units += weights[item, negative_visual_softmax[item]]

        negative_hidden_probabilities = sigmoid(hidden_units + hidden_biases)
        negative_hidden_states = negative_hidden_probabilities > numpy.random.random(negative_hidden_probabilities.shape)
        if not final_step:
            current_positive_hidden_states = negative_hidden_states
            neg_hidden_probs.fill(0)

    return negative_hidden_states, positive_hidden_states,  \
        [(item, negative_visual_softmax[item], rating) for (item, rating) in user_to_items[user]], \
        nrmse, ntrain


def updateWeights(emissions, EpsilonHB, EpsilonW, EpsilonVB, FEATURES, hidden_biases, hidden_biases_increment,
incremental_CD, ITEMS, Momentum, SOFTMAX, weight_cost, weights, visual_bias_increment, visible_biases):

    positive_CD = numpy.zeros((ITEMS, SOFTMAX, FEATURES))
    negative_CD = numpy.zeros((ITEMS, SOFTMAX, FEATURES))

    positive_hidden_activations = numpy.zeros(FEATURES)
    negative_hidden_activations = numpy.zeros(FEATURES)

    positive_visual_activations = numpy.zeros((ITEMS, SOFTMAX))
    negative_visual_activations = numpy.zeros((ITEMS, SOFTMAX))

    item_count = numpy.zeros(ITEMS)

    nrmse   = 0
    ntrain  = 0

    for emission in emissions:
        negative_hidden_states, positive_hidden_states, item_data, nrmse_update, ntrain_update = emission
        nrmse += nrmse_update
        ntrain += ntrain_update
        negative_hidden_activations += negative_hidden_states
        positive_hidden_activations += positive_hidden_states
        for (item, reconstructed_rating, rating) in item_data:
            item_count[item] += 1
            positive_visual_activations[item, rating] += 1
            negative_visual_activations[item, reconstructed_rating] += 1
            positive_CD[item, rating] += positive_hidden_states
            negative_CD[item, reconstructed_rating] += negative_hidden_states

    # batch update.
    numcases = len(emissions)

    # modify item counts to remove zeros before division.
    nonzero_item_count = item_count + (item_count == 0)

    # weight updates.
    positive_CD = numpy.divide(positive_CD.T, nonzero_item_count).T
    negative_CD = numpy.divide(negative_CD.T, nonzero_item_count).T

    activations = (positive_CD != 0) | (negative_CD != 0)
    incremental_CD *= (Momentum * activations + numpy.logical_not(activations))
    incremental_CD += numpy.multiply(activations, EpsilonW * ((positive_CD - negative_CD) - weight_cost * weights))

    weights += incremental_CD * activations

    # visual unit bias updates.
    positive_visual_activations = numpy.divide(positive_visual_activations.T, nonzero_item_count).T
    negative_visual_activations = numpy.divide(negative_visual_activations.T, nonzero_item_count).T

    activations = (positive_visual_activations != 0) | (negative_visual_activations != 0)
    visual_bias_increment *= (Momentum * activations + numpy.logical_not(activations))
    visual_bias_increment += numpy.multiply(activations, EpsilonVB * (positive_visual_activations - negative_visual_activations))

    visible_biases += visual_bias_increment * activations

    # hidden unit bias updates.
    positive_hidden_activations /= float(numcases)
    negative_hidden_activations /= float(numcases)

    activations = (positive_hidden_activations != 0) | (negative_hidden_activations != 0)
    hidden_biases_increment *= (Momentum * activations + numpy.logical_not(activations))
    hidden_biases_increment += numpy.multiply(activations, EpsilonHB * (positive_hidden_activations - negative_hidden_activations))

    hidden_biases += hidden_biases_increment * activations

    return nrmse, ntrain


def train(FEATURES, hidden_biases, ITEMS, SOFTMAX, user_to_items, visible_biases, weights):

    incremental_CD = numpy.zeros((ITEMS, SOFTMAX, FEATURES))
    hidden_biases_increment = numpy.zeros(FEATURES)
    visual_bias_increment = numpy.zeros((ITEMS, SOFTMAX))

    batch_size  = 1000
    EpsilonW    = 0.001
    EpsilonVB   = 0.008
    EpsilonHB   = 0.0006
    Momentum    = 0.8
    weight_cost = 0.0001

    nrmse       = 2.0
    last_rmse   = 10.0
    loop_count  = 0
    cd_steps    = 1

    while (nrmse < last_rmse) and loop_count < 80:

        if loop_count >= 10:
            cd_steps = 3 + (loop_count - 10) / 5.

        last_rmse = nrmse
        loop_count += 1
        ntrain = 0
        nrmse = 0

        if loop_count > 5:
            Momentum = 0.9

        user_keys = [user for user in user_to_items]
        number_of_batches = int(ceil(len(user_keys) / float(batch_size)))

        for batch in xrange(number_of_batches):
            pool = ThreadPool()
            startIndex = batch * batch_size
            endIndex = (batch + 1) * batch_size
            emissions = []
            for user in user_keys[startIndex : endIndex]:
            #    pool.apply_async(train_user, args=(cd_steps, FEATURES, hidden_biases, ITEMS, SOFTMAX,
            #        user, user_to_items, visible_biases, weights), callback=lambda x: emissions.append(x)
            #    )
                emissions.append(train_user(cd_steps, FEATURES, hidden_biases, ITEMS, SOFTMAX, user, user_to_items, visible_biases, weights))
            pool.close()
            pool.join()

            nrmse_update, ntrain_update = updateWeights(emissions, EpsilonHB, EpsilonW,
                EpsilonVB, FEATURES, hidden_biases, hidden_biases_increment,
                incremental_CD, ITEMS, Momentum, SOFTMAX, weight_cost, weights,
                visual_bias_increment, visible_biases)
            nrmse += nrmse_update
            ntrain += ntrain_update

        nrmse = (float(nrmse) / ntrain) ** 0.5
        print "Loop: {0:2}\tNRMSE: {1:5}".format(loop_count, nrmse)

        if loop_count > 8:
            EpsilonW *= 0.92
            EpsilonVB *= 0.92
            EpsilonHB *= 0.92
        elif loop_count > 6:
            EpsilonW *= 0.9
            EpsilonVB *= 0.9
            EpsilonHB *= 0.9
        elif loop_count > 2:
            EpsilonW *= 0.78
            EpsilonVB *= 0.78
            EpsilonHB *= 0.78


def sigmoid(x):
    return 1.0 / (1 + numpy.exp(-x))


def test(FEATURES, testing, hidden_biases, ITEMS, item_translation, labels, user_translation,
    user_to_items_training, visible_biases, weights):

  SOFTMAX = len(labels)

  user_to_items_testing = parse_input_file(testing, exclude_unseen=True, user_translation=user_translation, item_translation=item_translation, labels=labels)

  measures = dict((r, numpy.zeros(4, dtype=numpy.int)) for r in xrange(SOFTMAX))

  neg_hidden_probs = numpy.zeros((ITEMS, SOFTMAX))
  hidden_units = numpy.zeros(FEATURES)

  for user in user_to_items_training:
    neg_hidden_probs.fill(0)
    hidden_units.fill(0)
    for (item, rating) in user_to_items_training[user]:
      hidden_units += weights[item, rating]

    pos_hidden_probs = sigmoid(hidden_units + hidden_biases)

    for (item, _) in user_to_items_testing[user]:
      neg_hidden_probs[item] += numpy.dot(weights[item], pos_hidden_probs)
      neg_hidden_probs[item] = sigmoid(neg_hidden_probs[item] + visible_biases[item])

      normalization = sum(neg_hidden_probs[item])
      neg_hidden_probs[item] /= normalization

    for (item, rating) in user_to_items_testing[user]:
      prediction = sum(r * neg_hidden_probs[item, r] for r in xrange(SOFTMAX))

      prediction = int(round(prediction))

      if prediction == rating:
        measures[rating][0] += 1        # true positive.
      elif prediction != rating:
        measures[rating][2] += 1        # false negative.
        measures[prediction][1] += 1    # false positive
      measures[rating][3] += 1
  
  for label in measures:
    top = float(measures[label][0])
    recall_norm = (measures[label][0] + measures[label][2]) | 1
    precision_norm = (measures[label][0] + measures[label][1]) | 1
    count = measures[label][3]

    print "Label: {}".format(label + 1)
    print "Recall: {}".format(top / recall_norm)
    print "Precision: {}".format(top / precision_norm)
    print "Accuracy: {}".format(top / (count | 1))
    print "Count: {}".format(count)
    print

if __name__ == "__main__":
  import argparse
  start = time.time()
  parser = parser = argparse.ArgumentParser(description="RBM")
  parser.add_argument('-b', default=100, type=int, help="Batch size to use.")
  parser.add_argument('-f', default=18, type=int, help="Number of hidden features to use.")
  parser.add_argument('-s', required=False, action='store_true', help="Use seed for random number generation.")
  parser.add_argument('-v', required=False, action='store_true', help="Turn verbosity on.")

  args = vars(parser.parse_args())

  if args['s']:
    numpy.random.seed(42)

  hidden_biases, visible_biases, weights, item_translation, ITEMS, labels, user_translation, user_to_items = \
    run('./training',hidden_units=args['f'], verbose=args['v'], batch_size=args['b'])

  print "TRAINING ERROR:"
  test(args['f'], './training', hidden_biases, ITEMS, item_translation, labels, user_translation, user_to_items, visible_biases, weights)

  print "TESTING ERROR:"
  test(args['f'], './testing', hidden_biases, ITEMS, item_translation, labels, user_translation, user_to_items, visible_biases, weights)

  end = time.time()
  print "Runtime: {} seconds".format(end - start)
