package commons;

import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.lang3.tuple.Pair;

public abstract class Metrics {

  /**
   * Modify the rating, which we assume is the third value.
   * @param predictions
   */
  public static void collapse(List<Triplet> input, Set<Integer> labels,
      boolean verbose) {

    if (verbose) {
      System.out.println("Clipping ratings.");
    }

    int minLabel = Collections.min(labels);
    for (Triplet triplet : input) {

      triplet.third = triplet.third > minLabel ? minLabel + 1 : minLabel;
    }

  }

  /**
   * Get weights of labels.
   * These are the normalized inverse frequencies of each label
   * in the training data.
   * @param input
   * @return
   */
  public static Map<Integer, Double> getWeightsOfLabels(
      Map<Integer, Double> etas) {

    Map<Integer, Double> weights = new HashMap<>();

    int minimumKey = Collections.min(etas.keySet());

    double sum = 0;
    for (int key : etas.keySet()) {
      if (key == minimumKey) {
        weights.put(key, etas.get(key));
      } else {
        sum += etas.get(key);
      }
    }

    weights.put(Collections.min(weights.keySet()) + 1, sum);

    return weights;
  }

  /**
   * Report the score for our model. Points are awarded based on the inverse
   * frequency of the label correctly predicted.
   * We implicitly assume that the input and predictions are ordered.
   * @param inputAndPredictions
   * @param labelWeights
   */
  public static void evaluatePointsScored(List<Triplet> input,
      List<Pair<Double, Integer>> predictions, Map<Integer, Double> labelWeights) {
    double points = 0;
    double maxPoints = 0;

    assert (input.size() == predictions.size());

    Iterator<Triplet> inputIterator = input.iterator();
    Iterator<Pair<Double, Integer>> predictionIterator = predictions.iterator();

    while (inputIterator.hasNext()) {
      int groundTruth = inputIterator.next().third;
      int prediction = predictionIterator.next().getRight();
      if (prediction == groundTruth) {
        points += labelWeights.get(prediction);
      }
      maxPoints += labelWeights.get(groundTruth);
    }

    System.out.printf("Score:\t\t%f\n", points);
    System.out.printf("Max Possible:\t%f\n", maxPoints);
  }

  /**
   * Metrics are Accuracy/Precision/Recall per Label.
   * @param testingData
   */
  public static void evaluatePrecisionAndRecall(List<Pair<Double, Integer>> predictions,
      List<Integer> input) {

    assert (predictions.size() == input.size());

    Map<Integer, int[]> metrics = new HashMap<>();

    final int TRUE_POSITIVE = 0;
    final int FALSE_POSITIVE = 1;
    final int FALSE_NEGATIVE = 2;
    final int COUNT = 3;

    Iterator<Pair<Double, Integer>> predictionIter = predictions.iterator();
    Iterator<Integer> inputIter = input.iterator();

    while (inputIter.hasNext()) {

      int bucketedPrediction = predictionIter.next().getRight();
      int groundTruth = inputIter.next();

      if (!metrics.containsKey(bucketedPrediction)) {
        metrics.put(bucketedPrediction, new int[4]);
      }
      if (!metrics.containsKey(groundTruth)) {
        metrics.put(groundTruth, new int[4]);
      }

      if (bucketedPrediction == groundTruth) {
        metrics.get(groundTruth)[TRUE_POSITIVE] += 1;
      } else {
        metrics.get(groundTruth)[FALSE_NEGATIVE] += 1;
        metrics.get(bucketedPrediction)[FALSE_POSITIVE] += 1;
      }
      metrics.get(groundTruth)[COUNT] += 1;

    }

    for (int label : metrics.keySet()) {

      int[] data = metrics.get(label);

      int truePositives = data[TRUE_POSITIVE];

      double precision = ((double) truePositives)
          / ((truePositives + data[FALSE_POSITIVE]) | 1);
      double recall = ((double) truePositives)
          / ((truePositives + data[FALSE_NEGATIVE]) | 1);
      double accuracy = ((double) truePositives) / (data[COUNT] | 1);

      System.out.println("Label:\t\t" + label);
      System.out.println("Accuracy:\t" + accuracy);
      System.out.println("Precision:\t" + precision);
      System.out.println("Recall:\t\t" + recall);
      System.out.println("Count:\t\t" + data[COUNT]);
      System.out.println();
    }

  }

}
