import numpy
import time

from collections import defaultdict
from math import ceil


class RBM(object):

    def __init__(self, training_file, hidden_units, verbose=False, batch_size=100):
        self.epsilonw = 0.001
        self.epsilonvb = 0.008
        self.epsilonhb = 0.0006
        self.weight_cost = 0.0001
        self.momentum = 0.8
        self.final_momentum = 0.9
        self.batch_size = batch_size

        self.item_translation = None
        self.labels = None
        self.user_translation = None

        self.user_to_items = self.parse_input_file(training_file)
        self.ITEMS = len(self.item_translation)
        self.FEATURES = hidden_units
        self.SOFTMAX = len(self.labels)

        self.item_rating_count = numpy.zeros((self.ITEMS, self.SOFTMAX))

        self.visible_biases = numpy.zeros((self.ITEMS, self.SOFTMAX))
        self.hidden_biases = numpy.zeros(self.FEATURES)

        #self.weights = 0.02 * numpy.random.randn(self.ITEMS, self.SOFTMAX, self.FEATURES) - 0.01
        self.weights = numpy.zeros((self.ITEMS, self.SOFTMAX, self.FEATURES)) 

        self.populate_with_input_data()

        if verbose:
            print "BATCH SIZE:\t{}".format(self.batch_size)
            print "FEATURES:\t{}".format(self.FEATURES)
            print "LABELS:\t\t{}".format(self.SOFTMAX)
            print "ITEMS:\t\t{}".format(self.ITEMS)

    def populate_with_input_data(self):

        for user in self.user_to_items:
            for (item, rating) in self.user_to_items[user]:
                self.item_rating_count[item, rating] += 1

        non_zero_item_rating_count = self.item_rating_count.copy()
        non_zero_item_rating_count += (non_zero_item_rating_count == 0)

        normalization = self.item_rating_count.sum(axis=1)
        normalization += (normalization == 0)

        numpy.divide(non_zero_item_rating_count.T, normalization, self.visible_biases.T)
        numpy.log(self.visible_biases, self.visible_biases)

        activations = (self.item_rating_count > 0)

        self.visible_biases *= activations


    def parse_input_file(self, file, exclude_unseen=False):

        user_to_items = defaultdict(set)

        if self.item_translation is None:
            self.item_translation = dict()

        if self.labels is None:
            self.labels = set()

        if self.user_translation is None:
            self.user_translation = dict()

        with open(file, 'rb') as f:
            for line in f:
                user, item, rating = line.split()

                if user not in self.user_translation and not exclude_unseen:
                    self.user_translation[user] = len(self.user_translation)

                if rating not in self.labels and not exclude_unseen:
                    self.labels.add(rating)

                if item not in self.item_translation and not exclude_unseen:
                    self.item_translation[item] = len(self.item_translation)

                if item in self.item_translation and rating in self.labels and user in self.user_translation:
                    user_to_items[self.user_translation[user]].add((self.item_translation[item], int(rating) - 1))

        return user_to_items


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
#        positive_hidden_states = pos_hidden_probs > numpy.random.random(pos_hidden_probs.shape)
        positive_hidden_states = pos_hidden_probs > 0.5 
        current_positive_hidden_states = positive_hidden_states

        # first step : error calculation. NRMSE calculation.
        for (item, rating) in self.user_to_items[user]:
            nvp2[item] += numpy.dot(self.weights[item], pos_hidden_probs.T)
            self.sigmoid(nvp2[item] + self.visible_biases[item], nvp2[item])

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
                self.sigmoid(neg_hidden_probs[item] + self.visible_biases[item], neg_hidden_probs[item])

                normalization = sum(neg_hidden_probs[item])
                neg_hidden_probs[item] /= normalization

                #rand_value = numpy.random.rand()
                rand_value=0.5
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
#            negative_hidden_states = negative_hidden_probabilities > numpy.random.random(negative_hidden_probabilities.shape)
            negative_hidden_states = negative_hidden_probabilities > 0.5 
            if not final_step:
                current_positive_hidden_states = negative_hidden_states
                neg_hidden_probs.fill(0)

        return negative_hidden_states, positive_hidden_states,  \
            [(item, negative_visual_softmax[item], rating) for (item, rating) in self.user_to_items[user]], \
            nrmse, ntrain

    def updateWeights(self, emissions, incremental_CD, hidden_biases_increment,
        visual_bias_increment, EpsilonW, EpsilonVB, EpsilonHB, Momentum, positive_CD, negative_CD,
        positive_hidden_activations, negative_hidden_activations, positive_visual_activations, negative_visual_activations, item_count):

        positive_CD.fill(0)
        negative_CD.fill(0)
        
        positive_hidden_activations.fill(0)
        negative_hidden_activations.fill(0)
        
        positive_visual_activations.fill(0)
        negative_visual_activations.fill(0)

        item_count.fill(0)

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
        numpy.divide(positive_CD.T, nonzero_item_count, positive_CD.T)
        numpy.divide(negative_CD.T, nonzero_item_count, negative_CD.T)

        activations = (positive_CD != 0) | (negative_CD != 0)
        incremental_CD *= (Momentum * activations + numpy.logical_not(activations))
        incremental_CD += numpy.multiply(activations, EpsilonW * ((positive_CD - negative_CD) - self.weight_cost * self.weights))

        self.weights += incremental_CD * activations

        # visual unit bias updates.
        numpy.divide(positive_visual_activations.T, nonzero_item_count, positive_visual_activations.T)
        numpy.divide(negative_visual_activations.T, nonzero_item_count, negative_visual_activations.T)

        activations = (positive_visual_activations != 0) | (negative_visual_activations != 0)
        visual_bias_increment *= (Momentum * activations + numpy.logical_not(activations))
        visual_bias_increment += numpy.multiply(activations, EpsilonVB * (positive_visual_activations - negative_visual_activations))

        self.visible_biases += visual_bias_increment * activations

        # hidden unit bias updates.
        positive_hidden_activations /= float(numcases)
        negative_hidden_activations /= float(numcases)

        activations = (positive_hidden_activations != 0) | (negative_hidden_activations != 0)
        hidden_biases_increment *= (Momentum * activations + numpy.logical_not(activations))
        hidden_biases_increment += numpy.multiply(activations, EpsilonHB * (positive_hidden_activations - negative_hidden_activations))

        self.hidden_biases += hidden_biases_increment * activations

        return nrmse, ntrain

    def train(self):
        incremental_CD = numpy.zeros((self.ITEMS, self.SOFTMAX, self.FEATURES))
        hidden_biases_increment = numpy.zeros(self.FEATURES)
        visual_bias_increment = numpy.zeros((self.ITEMS, self.SOFTMAX))

        nvp2 = numpy.zeros((self.ITEMS, self.SOFTMAX))
        neg_hidden_probs = numpy.zeros((self.ITEMS, self.SOFTMAX))
        negative_visual_softmax = numpy.zeros(self.ITEMS)

        positive_CD = numpy.zeros((self.ITEMS, self.SOFTMAX, self.FEATURES))
        negative_CD = numpy.zeros((self.ITEMS, self.SOFTMAX, self.FEATURES))

        positive_hidden_activations = numpy.zeros(self.FEATURES)
        negative_hidden_activations = numpy.zeros(self.FEATURES)

        positive_visual_activations = numpy.zeros((self.ITEMS, self.SOFTMAX))
        negative_visual_activations = numpy.zeros((self.ITEMS, self.SOFTMAX))

        item_count = numpy.zeros(self.ITEMS)

        hidden_units = numpy.zeros(self.FEATURES)
        pos_hidden_probs = numpy.zeros(self.FEATURES)

        EpsilonW    = self.epsilonw
        EpsilonVB   = self.epsilonvb
        EpsilonHB   = self.epsilonhb
        Momentum    = self.momentum

        nrmse       = 2.0
        last_rmse   = 10.0
        loop_count  = 0
        cd_steps    = 1

        while (nrmse < last_rmse) and loop_count < 80:
            start_time = time.time()
            if loop_count >= 10:
                cd_steps = 3 + (loop_count - 10) / 5.

            last_rmse = nrmse
            loop_count += 1
            ntrain = 0
            nrmse = 0

            if loop_count > 5:
                Momentum = self.final_momentum

            user_keys = [user for user in self.user_to_items]
            number_of_batches = int(ceil(len(user_keys) / float(self.batch_size)))

            for batch in xrange(number_of_batches):
                startIndex = batch * self.batch_size
                endIndex = (batch + 1) * self.batch_size
                emissions = []
                for user in user_keys[startIndex : endIndex]:
                    emissions.append(self.train_user(user, cd_steps, neg_hidden_probs,
                        negative_visual_softmax, nvp2, hidden_units, pos_hidden_probs))
                nrmse_update, ntrain_update = self.updateWeights(emissions, incremental_CD,
                    hidden_biases_increment, visual_bias_increment,
                    EpsilonW, EpsilonVB, EpsilonHB, Momentum, positive_CD, negative_CD, positive_hidden_activations, negative_hidden_activations,
                    positive_visual_activations, negative_visual_activations, item_count)

                nrmse += nrmse_update
                ntrain += ntrain_update

            nrmse = (float(nrmse) / ntrain) ** 0.5
            end_time = time.time()
            print "Loop: {0:2}\tNRMSE: {1:5}\tTime:{2} secs".format(loop_count, nrmse, end_time - start_time)

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


    def sigmoid(self, x, y):
        numpy.exp(x, y)
        y /= (1 + y)
        return y


    def test(self, testing):
        if type(testing) is str:
            users_to_items_testing = self.parse_input_file(testing, exclude_unseen=True)
            self._test(users_to_items_testing)
        elif type(testing) is defaultdict:
            self._test(testing)


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
                self.sigmoid(neg_hidden_probs[item] + self.visible_biases[item], neg_hidden_probs[item])

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
    parser = parser = argparse.ArgumentParser(description="RBM")
    parser.add_argument('-b', default=100, type=int, help="Batch size to use.")
    parser.add_argument('-f', default=18, type=int, help="Number of hidden features to use.")
    parser.add_argument('-s', required=False, action='store_true', help="Use seed for random number generation.")
    parser.add_argument('-v', required=False, action='store_true', help="Turn verbosity on.")

    args = vars(parser.parse_args())

    if args['s']:
        numpy.random.seed(42)

    r = RBM('./training_small', hidden_units=args['f'], verbose=args['v'], batch_size=args['b'])
    r.train()

    print "TRAINING ERROR:"
    r.test(r.user_to_items)

    print "TESTING ERROR:"
    r.test('./testing_small')

    end = time.time()
    print "Runtime: {} seconds".format(end - start)
