package Regression;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math.stat.regression.OLSMultipleLinearRegression;

public class LinearRegression {

  double eta;
  double[] weights;

  public LinearRegression(double eta) {
    this.eta = eta;
  }

  public Pair<double[][], double[]> parseFile(String filename) {
    try (Scanner fileScanner = new Scanner(new File(filename))) {
      List<String> lines = new ArrayList<String>();
      while (fileScanner.hasNext()) {
        lines.add(fileScanner.nextLine());
      }
      double[][] datax = new double[lines.size()][];
      double[] datay = new double[lines.size()];
      for (int i = 0; i < lines.size(); ++i) {
        String[] tokens = lines.get(i).split(",");
        double[] x = new double[tokens.length - 1];
        double y = Double.parseDouble(tokens[tokens.length - 1]);
        for (int j = 0; j < tokens.length - 1; ++j) {
          x[j] = Double.parseDouble(tokens[j]);
        }
        datax[i] = x;
        datay[i] = y;
      }
      return Pair.of(datax, datay);
    } catch (Exception e) {
      e.printStackTrace();
      System.exit(1);
    }
    return null;
  }

  public void train(String filename) {
    Pair<double[][], double[]> trainData = parseFile(filename);
    OLSMultipleLinearRegression regression = new OLSMultipleLinearRegression();
    regression.newSampleData(trainData.getRight(), trainData.getLeft());
    weights = regression.estimateRegressionParameters();
    for (int i = 0; i < weights.length; ++i) {
      System.out.printf("w%d=%f, ", i, weights[i]);
    }
    System.out.println();
  }

  public void test(String filename) {
    double totalError = 0;
    Pair<double[][], double[]> testData = parseFile(filename);
    double[][] datax = testData.getLeft();
    double[] datay = testData.getRight();
    for (int i = 0; i < datax.length; ++i) {
      double[] xi = datax[i];
      double yi = datay[i];
      double guess = predict(xi, weights);
      totalError += Math.abs(yi - guess);
    }
    System.out.printf("Total Error: %f\n", totalError);
  }

  public double predict(double[] x, double[] wieghts) {
    double total = weights[0];
    for (int i = 0; i < x.length; ++i) {
      total += x[i] * wieghts[i + 1];
    }
    return total;
  }

  public static void main(String[] args) {
    LinearRegression lr = new LinearRegression(0.01);
    lr.train("data/lrtest.csv");
    lr.test("data/lrtest.csv");
  }
}
