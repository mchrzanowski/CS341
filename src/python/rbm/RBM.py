import numpy

from collections import defaultdict

class RBM(object):

    def __init__(self, training_file, labels, hidden_units):
        
        self.item_translation = None
        self.labels = None
        self.user_translation = None

        self.user_to_items = self.parse_input_file(training_file)
        self.ITEMS = len(self.item_translation)
        self.FEATURES = hidden_units
        self.SOFTMAX = len(self.labels)

        self.pos_hidden_probs = numpy.zeros(self.FEATURES)
        self.neg_hidden_probs = numpy.zeros((self.ITEMS, self.SOFTMAX))

        self.item_rating_count = numpy.zeros((self.ITEMS, self.SOFTMAX))
        self.item_count = numpy.zeros(self.ITEMS)

        self.visible_biases = numpy.zeros((self.ITEMS, self.SOFTMAX))
        self.hidden_biases = numpy.zeros(self.FEATURES)

        self.weights = 0.02 * numpy.random.randn(self.ITEMS, self.SOFTMAX, self.FEATURES) - 0.01

        self.score_setup()
        self.initialize()


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


    def score_setup(self):
        for user in self.user_to_items:
            for (item, rating) in self.user_to_items[user]:
                self.item_rating_count[item, rating] += 1


    def initialize(self):
        for i in xrange(self.ITEMS):
            item_total = sum(self.item_rating_count[i])
            for r in xrange(self.SOFTMAX):
                if self.item_rating_count[i, r] > 0:
                    self.visible_biases[i, r] = numpy.log(float(self.item_rating_count[i, r]) / item_total)


    def train(self):

        epsilonw = 0.001
        epsilonvb = 0.008
        epsilonhb = 0.0006

        weight_cost = 0.0001
        momentum = 0.8
        final_momentum = 0.9

        batch_size = 100

        positive_CD = numpy.zeros((self.ITEMS, self.SOFTMAX, self.FEATURES))
        negative_CD = numpy.zeros((self.ITEMS, self.SOFTMAX, self.FEATURES))
        incremental_CD = numpy.zeros((self.ITEMS, self.SOFTMAX, self.FEATURES))

        positive_hidden_states = numpy.zeros(self.FEATURES)
        current_positive_hidden_states = numpy.zeros(self.FEATURES)
        positive_hidden_activations = numpy.zeros(self.FEATURES)
        negative_hidden_activations = numpy.zeros(self.FEATURES)
        negative_hidden_probabilities = numpy.zeros(self.FEATURES)
        negative_hidden_states = numpy.zeros(self.FEATURES)

        hidden_biases_increment = numpy.zeros(self.FEATURES)

        nvp2 = numpy.zeros((self.ITEMS, self.SOFTMAX))
        negative_visual_softmax = numpy.zeros(self.ITEMS)

        positive_visual_activations = numpy.zeros((self.ITEMS, self.SOFTMAX))
        negative_visual_activations = numpy.zeros((self.ITEMS, self.SOFTMAX))
        visual_bias_increment = numpy.zeros((self.ITEMS, self.SOFTMAX))

        EpsilonW = epsilonw
        EpsilonVB = epsilonvb
        EpsilonHB = epsilonhb
        Momentum = momentum

        cd_steps = 1

        nrmse = 2.0
        last_rmse = 10.0
        loop_count = 0

        while (nrmse < last_rmse) and loop_count < 80:
            
            if loop_count >= 10:
                cd_steps = 3 + (loop_count - 10) / 5.

            last_rmse = nrmse
            loop_count += 1
            ntrain = 0
            nrmse = 0

            if loop_count > 5:
                Momentum = final_momentum

            positive_CD.fill(0)
            negative_CD.fill(0)
            positive_hidden_activations.fill(0)
            negative_hidden_activations.fill(0)
            positive_visual_activations.fill(0)
            negative_visual_activations.fill(0)
            self.item_count.fill(0)

            for user_number, user in enumerate(self.user_to_items):

                self.neg_hidden_probs.fill(0)
                nvp2.fill(0)

                hidden_units = numpy.zeros(self.FEATURES)

                for (item, rating) in self.user_to_items[user]:
                    self.item_count[item] += 1
                    positive_visual_activations[item, rating] += 1.0
                    for feature in xrange(self.FEATURES):
                        hidden_units[feature] += self.weights[item, rating, feature]

                for feature in xrange(self.FEATURES):
                    self.pos_hidden_probs[feature] = self.sigmoid(hidden_units[feature] + self.hidden_biases[feature])

                    if self.pos_hidden_probs[feature] > numpy.random.rand():
                        positive_hidden_states[feature] = 1
                        positive_hidden_activations[feature] += 1.0
                    else:
                        positive_hidden_states[feature] = 0

                for feature in xrange(self.FEATURES):
                    current_positive_hidden_states[feature] = positive_hidden_states[feature]

                step = 0
                while step < cd_steps:
                    step += 1
                    final_step = step >= cd_steps

                    for (item, rating) in self.user_to_items[user]:
                        for feature in xrange(self.FEATURES):
                            if current_positive_hidden_states[feature] == 1:
                                for r in xrange(self.SOFTMAX):
                                    self.neg_hidden_probs[item, r] += self.weights[item, r, feature]

                            if step == 1:
                                for r in xrange(self.SOFTMAX):
                                    nvp2[item, r] += self.pos_hidden_probs[feature] * self.weights[item, r, feature]

                        for r in xrange(self.SOFTMAX):
                            self.neg_hidden_probs[item, r] = self.sigmoid(self.neg_hidden_probs[item, r] + self.visible_biases[item, r])

                        normalization = sum(self.neg_hidden_probs[item])
                        if normalization != 0:
                            for r in xrange(self.SOFTMAX):
                                self.neg_hidden_probs[item, r] /= float(normalization)

                        if step == 1:
                            for r in xrange(self.SOFTMAX):
                                nvp2[item, r] = self.sigmoid(nvp2[item, r] + self.visible_biases[item, r])
                            normalization = sum(nvp2[item])
                            if normalization != 0:
                                for r in xrange(self.SOFTMAX):
                                    nvp2[item, r] /= float(normalization)

                        rand_value = numpy.random.rand()
                        for r in xrange(self.SOFTMAX):
                            rand_value -= self.neg_hidden_probs[item, r]
                            if rand_value <= 0:
                                negative_visual_softmax[item] = r
                                break
                        if rand_value > 0:
                            negative_visual_softmax[item] = self.SOFTMAX - 1

                        if final_step:
                            negative_visual_activations[item, negative_visual_softmax[item]] += 1.0

                    hidden_units.fill(0)

                    for (item, rating) in self.user_to_items[user]:
                        for feature in xrange(self.FEATURES):
                            hidden_units[feature] += self.weights[item, negative_visual_softmax[item], feature]

                    for feature in xrange(self.FEATURES):
                        negative_hidden_probabilities[feature] = self.sigmoid(hidden_units[feature] + self.hidden_biases[feature])

                        if negative_hidden_probabilities[feature] > numpy.random.rand():
                            negative_hidden_states[feature] = 1
                            if final_step:
                                negative_hidden_activations[feature] += 1.0
                        else:
                            negative_hidden_states[feature] = 0

                    if step == 1:
                        for (item, rating) in self.user_to_items[user]:
                            expectedV = 0
                            for r in xrange(self.SOFTMAX):
                                expectedV += r * nvp2[item, r]
                            vdelta = rating - expectedV
                            nrmse += vdelta ** 2
                        
                        ntrain += len(self.user_to_items[user])


                    if not final_step:
                        for feature in xrange(self.FEATURES):
                            current_positive_hidden_states[feature] = negative_hidden_states[feature]
                        self.neg_hidden_probs.fill(0)

                for (item, rating) in self.user_to_items[user]:
                    for feature in xrange(self.FEATURES):
                        if positive_hidden_states[feature] == 1:
                            positive_CD[item, rating, feature] += 1.0

                        negative_CD[item, negative_visual_softmax[item], feature] += negative_hidden_states[feature]

                if (user_number + 1) % batch_size == 0 or (user_number + 1) == len(self.user_to_items):
                    numcases = user_number % batch_size
                    numcases += 1

                    for item in xrange(self.ITEMS):
                        if self.item_count[item] == 0:
                            continue
                        for feature in xrange(self.FEATURES):
                            for r in xrange(self.SOFTMAX):
                                CDp = positive_CD[item, r, feature]
                                CDn = negative_CD[item, r, feature]
                                if CDp != 0 or CDn != 0:
                                    CDp /= float(self.item_count[item])
                                    CDn /= float(self.item_count[item])

                                    incremental_CD[item, r, feature] = Momentum * incremental_CD[item, r, feature] + \
                                        EpsilonW * ((CDp - CDn) - weight_cost * self.weights[item, r, feature])
                                    self.weights[item, r, feature] += incremental_CD[item, r, feature]
                        
                        for r in xrange(self.SOFTMAX):
                            if positive_visual_activations[item, r] != 0 or negative_visual_activations[item, r] != 0:
                                positive_visual_activations[item, r] /= float(self.item_count[item])
                                negative_visual_activations[item, r] /= float(self.item_count[item])
                                visual_bias_increment[item, r] = Momentum * visual_bias_increment[item, r] + \
                                    EpsilonVB * ((positive_visual_activations[item, r] - negative_visual_activations[item, r]))
                                self.visible_biases[item, r] += visual_bias_increment[item, r]

                    for feature in xrange(self.FEATURES):
                        if positive_hidden_activations[feature] != 0 or negative_hidden_activations[feature] != 0:
                            positive_hidden_activations[feature] /= float(numcases)
                            negative_hidden_activations[feature] /= float(numcases)
                            hidden_biases_increment[feature] = Momentum * hidden_biases_increment[feature] + \
                                EpsilonHB * ((positive_hidden_activations[feature] - negative_hidden_activations[feature]))
                            self.hidden_biases[feature] += hidden_biases_increment[feature]

                    positive_CD.fill(0)
                    negative_CD.fill(0)
                    positive_hidden_activations.fill(0)
                    negative_hidden_activations.fill(0)
                    positive_visual_activations.fill(0)
                    negative_visual_activations.fill(0)
                    self.item_count.fill(0)

            nrmse = (float(nrmse) / ntrain) ** 0.5
            print "Loop: {}\tNRMSE: {}".format(loop_count, nrmse)

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


    def sigmoid(self, x):
        return 1.0 / (1 + numpy.exp(-x))


    def test(self, user_to_items_training, user_to_items_testing):

        accuracy = 0
        counter = 0

        for user in user_to_items_training:
            self.neg_hidden_probs.fill(0)
            hidden_units = numpy.zeros(self.FEATURES)
            for (item, rating) in user_to_items_training[user]:
                for feature in xrange(self.FEATURES):
                    hidden_units[feature] += self.weights[item, rating, feature]

            for feature in xrange(self.FEATURES):
                self.pos_hidden_probs[feature] = self.sigmoid(hidden_units[feature] + self.hidden_biases[feature])

            for (item, _) in user_to_items_testing[user]:
                for feature in xrange(self.FEATURES):
                    for r in xrange(self.SOFTMAX):
                        self.neg_hidden_probs[item, r] += self.pos_hidden_probs[feature] * self.weights[item, r, feature]
                for r in xrange(self.SOFTMAX):
                    self.neg_hidden_probs[item, r] = self.sigmoid(self.neg_hidden_probs[item, r] + self.visible_biases[item, r])

                normalization = sum(self.neg_hidden_probs[item])
                if normalization != 0:
                    self.neg_hidden_probs[item] /= float(normalization)

            for (item, rating) in user_to_items_testing[user]:
                prediction = sum(r * self.neg_hidden_probs[item, r] for r in xrange(self.SOFTMAX))
                prediction = round(prediction)

                if prediction == rating:
                    accuracy += 1
                counter += 1

        print "Accuracy: {}".format(float(accuracy) / counter)


if __name__ == "__main__":
    import time
    numpy.random.seed(42)
    start = time.time()
    r = RBM('./training', labels=3, hidden_units=18)
    r.train()
    r.test(r.user_to_items, r.user_to_items)
    end = time.time()
    print "Runtime: {} seconds".format(end - start)

