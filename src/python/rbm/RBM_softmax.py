import numpy
import time

from RBM_LA import RBM
from collections import defaultdict
from math import ceil


class RBM_softmax(RBM):

    def __init__(self, training_file, hidden_units, verbose=False, batch_size=100):
        RBM.__init__(self, training_file, hidden_units, verbose, batch_size)

    def train_user(self, user, cd_steps, neg_hidden_probs, negative_visual_softmax, nvp2, hidden_units, pos_hidden_probs):

        nvp2.fill(0)
        neg_hidden_probs.fill(0)
        negative_visual_softmax.fill(0)
        hidden_units.fill(0)
        pos_hidden_probs.fill(0)

        nrmse = 0
        ntrain = 0

        for (item, rating) in self.user_to_items[user]:
            hidden_units += self.weights[item, rating]

        self.sigmoid(hidden_units + self.hidden_biases, pos_hidden_probs)
        # sample binary values.
        positive_hidden_states = pos_hidden_probs > numpy.random.random(pos_hidden_probs.shape)
        current_positive_hidden_states = positive_hidden_states

        # first step : error calculation. NRMSE calculation.
        for (item, rating) in self.user_to_items[user]:
            nvp2[item] += numpy.dot(self.weights[item], pos_hidden_probs.T)
            numpy.exp(nvp2[item] + self.visible_biases[item], nvp2[item])

            normalization = sum(nvp2[item])
            nvp2[item] /= normalization

            expectedV = sum(r * nvp2[item, r] for r in xrange(self.SOFTMAX))
            vdelta = rating - expectedV
            nrmse += vdelta ** 2                                                    # EMIT THIS.

        ntrain += len(self.user_to_items[user])                                     # EMIT THIS.

        step = 0
        while step < cd_steps:
            step += 1
            final_step = step >= cd_steps

            # POSITIVE CD calculation
            # AKA reconstruct the visible units
            for (item, _) in self.user_to_items[user]:
                neg_hidden_probs[item] += numpy.dot(self.weights[item], current_positive_hidden_states.T)
                numpy.exp(neg_hidden_probs[item] + self.visible_biases[item], neg_hidden_probs[item])

                normalization = sum(neg_hidden_probs[item])
                neg_hidden_probs[item] /= normalization

                rand_value = numpy.random.rand()
                for r in xrange(self.SOFTMAX):
                    rand_value -= neg_hidden_probs[item, r]
                    if rand_value <= 0:
                        negative_visual_softmax[item] = r
                        break
                if rand_value > 0:
                    negative_visual_softmax[item] = self.SOFTMAX - 1

            # NEGATIVE CD calculation.
            # AKA create the hidden layer
            hidden_units.fill(0)
            for (item, _) in self.user_to_items[user]:
                hidden_units += self.weights[item, negative_visual_softmax[item]]

            negative_hidden_probabilities = numpy.zeros(self.FEATURES)
            self.sigmoid(hidden_units + self.hidden_biases, negative_hidden_probabilities)
            # sample binary values.
            negative_hidden_states = negative_hidden_probabilities > numpy.random.random(negative_hidden_probabilities.shape)
            if not final_step:
                current_positive_hidden_states = negative_hidden_states
                neg_hidden_probs.fill(0)

        return negative_hidden_states, positive_hidden_states,  \
            [(item, negative_visual_softmax[item], rating) for (item, rating) in self.user_to_items[user]], \
            nrmse, ntrain

    def _test(self, user_to_items_test):

        measures = dict((r, numpy.zeros(4, dtype=numpy.int)) for r in xrange(self.SOFTMAX))

        neg_hidden_probs = numpy.zeros((self.ITEMS, self.SOFTMAX))
        hidden_units = numpy.zeros(self.FEATURES)
        pos_hidden_probs = numpy.zeros(self.FEATURES)

        for user in self.user_to_items:
            neg_hidden_probs.fill(0)
            hidden_units.fill(0)
            pos_hidden_probs.fill(0)
            for (item, rating) in self.user_to_items[user]:
                hidden_units += self.weights[item, rating]

            self.sigmoid(hidden_units + self.hidden_biases, pos_hidden_probs)

            for (item, _) in user_to_items_test[user]:
                neg_hidden_probs[item] += numpy.dot(self.weights[item], pos_hidden_probs)
                numpy.exp(neg_hidden_probs[item] + self.visible_biases[item], neg_hidden_probs[item])

                normalization = sum(neg_hidden_probs[item])
                neg_hidden_probs[item] /= normalization

            for (item, rating) in user_to_items_test[user]:
                prediction = sum(r * neg_hidden_probs[item, r] for r in xrange(self.SOFTMAX))

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
    parser = parser = argparse.ArgumentParser(description="RBM_softmax")
    parser.add_argument('-b', default=100, type=int, help="Batch size to use.")
    parser.add_argument('-f', default=18, type=int, help="Number of hidden features to use.")
    parser.add_argument('-s', required=False, action='store_true', help="Use seed for random number generation.")
    parser.add_argument('-v', required=False, action='store_true', help="Turn verbosity on.")

    args = vars(parser.parse_args())

    if args['s']:
        numpy.random.seed(42)

    r = RBM_softmax('./training', hidden_units=args['f'], verbose=args['v'], batch_size=args['b'])
    r.train()

    print "TRAINING ERROR:"
    r.test(r.user_to_items)

    print "TESTING ERROR:"
    r.test('./testing')

    end = time.time()
    print "Runtime: {} seconds".format(end - start)
