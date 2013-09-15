package Regression;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

import org.apache.commons.lang3.tuple.Pair;

public class RegressionSGD {

  double eta;
  double[] weights;
  Random rnd;

  public RegressionSGD(double eta, int numParams) {
    this.eta = eta;
    this.weights = new double[] { 0.0, 0.5, 1.5, 3.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0}; 
    /*this.weights = new double[numParams + 1];
    this.rnd = new Random();
    for (int i = 0; i < this.weights.length; i++){
        this.weights[i] = this.rnd.nextGaussian();
    }*/
  }

  public void train(double[][] x, double guess, double theoretical) {
    double error = theoretical - guess;
    for(int index = 0; index < x.length; ++index) {
      double[] xi = x[index];
      weights[0] += eta * error * 0;
      for(int j = 1; j < weights.length; ++j) {
        weights[j] += eta * error * xi[j-1];
      }
    }
  }

  public void train(double[][] x, double[] y) {
    int[] permutation = getPermutation(x.length);
    for(int i = 0; i < x.length; ++i) {
      int index = permutation[i];
      double[] xi = x[index];
      double yi = y[index];
      double guess = predict(xi);
      double error = yi - guess;
      weights[0] += eta * error;
      for(int j = 1; j < weights.length; ++j) {
        weights[j] += eta * error * xi[j-1];
      }
    }
  }

  public void test(double[][] x, double[] y) {
    double totalError = 0;
    for (int i = 0; i < x.length; ++i) {
      double[] xi = x[i];
      double yi = y[i];
      double guess = predict(xi);
      totalError += Math.abs(yi - guess);
    }
    System.out.printf("Total Error: %f\n", totalError);
  }

  public double predict(double[] x) {
    double total = weights[0];
    for (int i = 0; i < x.length; ++i) {
      total += x[i] * weights[i + 1];
    }
    return total;
  }

  public void printWeights() {
    for (int i = 0; i < weights.length; ++i) {
      System.out.printf("w%d=%f, ", i, weights[i]);
    }
    System.out.println();    
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

  public int[] getPermutation(int length) {
    int[] perm = new int[length];
    for(int i = 0; i < perm.length; ++i) {
      perm[i] = i;
    }
    // Fisher Yates Shuffle
    for (int i = perm.length - 1; i >= 0; i--) {
      int index = rnd.nextInt(i + 1);
      int temp = perm[index];
      perm[index] = perm[i];
      perm[i] = temp;
    }
    return perm;
  }
  
  public static void main(String[] args) {
    RegressionSGD rsgd = new RegressionSGD(0.01, 2);
    Pair<double[][], double[]> train = rsgd.parseFile("data/lrtest.csv");    
    rsgd.train(train.getLeft(), train.getRight());
    rsgd.printWeights();
    rsgd.test(train.getLeft(), train.getRight());
  }
}

