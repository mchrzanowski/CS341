package Regression;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math.stat.regression.OLSMultipleLinearRegression;

import SVD.SVD;
import SVD.SVDTrainedModel;

import com.google.gson.Gson;
import commons.Metrics;

public class WalmartSearchRank {

  public class WalmartQuery {
    String visitorid;
    int[] shownitems;
    int[] clickeditems;

    public int find(int[] array, int value) {
      for (int i = 0; i < array.length; i++) {
        if (array[i] == value) {
          return i;
        }
      }
      return -1;
    }

    public double[] getRatings() {
      double[] ratings = new double[shownitems.length];
      Arrays.fill(ratings, 1.0);
      for (int i = 0; i < clickeditems.length; ++i) {
        int item = clickeditems[i];
        int index = find(shownitems, item);
        if (index >= 0)
          ratings[index] = 2.0;
      }
      return ratings;
    }
  }

  private final SVDTrainedModel userItemSVD;
  private final SVDTrainedModel userRankSVD;
  private final SVDTrainedModel itemRankSVD;
  private final Gson gson = new Gson();
  private double[] weights = { 0, 0.7, 0.15, 0.15 };

  public WalmartSearchRank(String userItemFile, String userRankFile,
      String itemRankFile) {
    userItemSVD = SVD.unserializeModel(new File(userItemFile));
    userRankSVD = SVD.unserializeModel(new File(userRankFile));
    itemRankSVD = SVD.unserializeModel(new File(itemRankFile));
  }

  private WalmartQuery parseQuery(String json) {
    return gson.fromJson(json, WalmartQuery.class);
  }

  public Pair<double[][], double[]> concat(
      List<Pair<double[][], double[]>> data, int totalLength) {
    double[][] x = new double[totalLength][];
    double[] y = new double[totalLength];
    int offset = 0;
    for (Pair<double[][], double[]> pair : data) {
      double[][] xi = pair.getLeft();
      double[] yi = pair.getRight();
      for (int i = 0; i < xi.length; ++i) {
        x[offset] = xi[i];
        y[offset] = yi[i];
        offset++;
      }
    }
    return Pair.of(x, y);
  }

  public void train(String filename) throws Exception {
    int totalLength = 0;
    List<Pair<double[][], double[]>> data = new ArrayList<>();
    try (Scanner fileScanner = new Scanner(new File(filename))) {
      while (fileScanner.hasNext()) {
        String line = fileScanner.nextLine();
        Pair<double[][], double[]> pair = computeRegressionInput(line);
        if (pair != null) {
          totalLength += pair.getLeft().length;
          data.add(pair);
        }
      }
    }
    System.out.println(data.size());
    Pair<double[][], double[]> input = concat(data, totalLength);
    data = null;
    OLSMultipleLinearRegression regression = new OLSMultipleLinearRegression();
    regression.newSampleData(input.getRight(), input.getLeft());
    weights = regression.estimateRegressionParameters();
    for (int i = 0; i < weights.length; ++i) {
      System.out.printf("w%d=%f, ", i, weights[i]);
    }
    System.out.println();
  }

  public void test(String filename) throws Exception {
    List<Pair<Double, Integer>> predictions = new ArrayList<>();
    List<Integer> truths = new ArrayList<Integer>();
    try (Scanner fileScanner = new Scanner(new File(filename))) {
      while (fileScanner.hasNext()) {
        String line = fileScanner.nextLine();
        Pair<double[][], double[]> data = computeRegressionInput(line);
        if (data == null)
          continue;
        double[][] datax = data.getLeft();
        double[] datay = data.getRight();
        for (int i = 0; i < datax.length; ++i) {
          double[] xi = datax[i];
          double yi = datay[i];
          double guess = predict(xi);
          guess = guess < 1 ? 1 : guess;
          guess = guess > 2 ? 2 : guess;
          predictions.add(Pair.of(guess, (int) Math.round(guess)));
          truths.add((int) Math.round(yi));
        }
      }
    }
    Metrics.evaluatePrecisionAndRecall(predictions, truths);
  }

  public double predict(double[] x) {
    double total = weights[0];
    for (int i = 0; i < x.length; ++i) {
      total += x[i] * weights[i + 1];
    }
    return total;
  }

  public Pair<double[][], double[]> computeRegressionInput(String query) {
    try {
      WalmartQuery wquery = parseQuery(query);
      String user = wquery.visitorid;
      int[] shownItems = wquery.shownitems;
      double[][] x = new double[shownItems.length][];
      for (int rank = 0; rank < shownItems.length; ++rank) {
        String item = "" + shownItems[rank];
        double rating1 = userItemSVD.predict(user, item).getLeft();
        double rating2 = itemRankSVD.predict(item, rank + "").getLeft();
        double rating3 = userRankSVD.predict(user, rank + "").getLeft();
        x[rank] = new double[] { rating1, rating2, rating3 };
      }
      double[] y = wquery.getRatings();
      return Pair.of(x, y);
    } catch (Exception e) {
      return null;
    }
  }

  public static void main(String[] args) throws Exception {
    String userItemFile = "models/UISVD.ser";
    String userRankFile = "models/URSVD.ser";
    String itemRankFile = "models/IRSVD.ser";
    WalmartSearchRank wsr = new WalmartSearchRank(userItemFile, userRankFile,
        itemRankFile);
    // wsr.train("data/query/training/good_training");
    wsr.test("data/query/testing/good_testing");
  }
}
