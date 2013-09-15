package Regression;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math.stat.regression.OLSMultipleLinearRegression;

import SVD.SVD;
import SVD.SVDTrainedModel;

import com.google.gson.Gson;

public class HackyWalmartSearchRank {

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
  private double[] weights = { 0, 3.0, 1.0, 2.0 };

  public HackyWalmartSearchRank(String userItemFile, String userRankFile,
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

  public void test(String filename) {
    double score = 0;
    int queries = 0;
    double oldScores = 0;
    double newScores = 0;
    double[] pdf = new double[] {0.241175721556,0.130981735144,
        	0.0892994129563,0.0713992135027,0.0570851106386,0.0494814922847,
        	0.0441505114279,0.0403570151304,0.0361282328143,0.033366522917,0.0309776498884,
        	0.0295540602239,0.0281801065337,0.0269866231653,0.0262979210055,0.0274620860907,
        	0.00267735045824,0.00261067851026,0.00254907391989,0.00251444697622,0.00241471655241,
        	0.00233489360492,0.00227514704567,0.00226715992487,0.00216866014504,0.00211856569553,
        	0.00208005377769,0.00205942239314,0.00208058464373,0.00205650262995,0.00206755429559,
        	0.00219426236611,2.53367880486e-05,2.45163587213e-05,2.35270174737e-05,2.34063661021e-05,
        	2.44439678983e-05,2.35994082967e-05,2.32615844561e-05,2.3985492686e-05,2.4709400916e-05,
        	2.29720211641e-05,2.16207258015e-05,2.32857147304e-05,2.26583275978e-05,2.35994082967e-05,
        	2.16931166245e-05,2.33339752791e-05,2.14518138812e-05,2.19344193678e-05,2.13070322352e-05,
        	2.24652854031e-05,2.10898597662e-05,2.08485570229e-05,2.11381203149e-05,2.21033312881e-05,
        	2.36717991197e-05,2.41785348807e-05,2.63019990219e-05,2.5071355031e-05};
    
    try (BufferedReader fileScanner = new BufferedReader(new FileReader(new File(filename)))) {
      String line = null;
      while ((line = fileScanner.readLine()) != null) {
        queries++;
        if (queries % 100000 == 0){
        	System.out.println(queries);
        }
        WalmartQuery wquery = null;
        try{
        	wquery = parseQuery(line);
        } catch (Exception e){
        	continue;
        }
        
        Pair<double[][], double[]> data = computeRegressionInput(line);
        List<Pair<Double, Integer>> predictions = new ArrayList<>();
        if (data == null)
          continue;
        double[][] datax = data.getLeft();
        for (int i = 0; i < datax.length; ++i) {
          double[] xi = datax[i];
          double guess = predict(xi);          
          
          predictions.add(Pair.of(guess, i));
        }
        Collections.sort(predictions, Collections.reverseOrder());
        Arrays.sort(wquery.clickeditems);
        
        assert predictions.size() == wquery.shownitems.length;
        
        double oldScore = 0;
        // change this ratio to only look at the top K%
        double PERCENT = 1.0;
        for (int i = 0; i < wquery.shownitems.length * PERCENT; i++){
          if (Arrays.binarySearch(wquery.clickeditems, wquery.shownitems[i]) >= 0){
            oldScore += pdf[i];
          }
        	
        }
        
        List<Pair<Double, Integer>> realPrediction = predictions;
        assert realPrediction.size() == wquery.shownitems.length;

        double newScore = 0;
        for (int i = 0; i < realPrediction.size() * PERCENT; i++){
        	int index = realPrediction.get(i).getRight();
    		if (Arrays.binarySearch(wquery.clickeditems, wquery.shownitems[index]) >= 0){
    			newScore += pdf[i];
    		}
        }
        
    	score += newScore - oldScore;
        
        oldScores += oldScore;
        newScores += newScore;
        
      }
    } catch (Exception e){
  	  System.out.println("HERE 2");
    }
    
    System.out.println("Score: " + score);
    System.out.println("New Score: " + newScores);
    System.out.println("Old Score: " + oldScores);
    System.out.println("Queries: " + queries);
    System.out.println("Percent Improvement: " + 100 * ((newScores - oldScores) / oldScores));
    
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
    String userItemFile = "/Users/polak/Documents/workspace/CS341/trained_models/user_item_3week";
    String userRankFile = "/Users/polak/Documents/workspace/CS341/trained_models/user_rank_3week";
    String itemRankFile = "/Users/polak/Documents/workspace/CS341/trained_models/item_rank_3week";
    HackyWalmartSearchRank wsr = new HackyWalmartSearchRank(userItemFile, userRankFile,
        itemRankFile);
    // wsr.train("data/query/training/good_training");
    //wsr.test("/Users/polak/Documents/workspace/CS341/input_data/query/testing/good_testing");
    wsr.test("/Users/polak/Documents/workspace/CS341/input_data/query/training/good_training");
  }
}