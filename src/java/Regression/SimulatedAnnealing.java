package Regression;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.Options;
import org.apache.commons.lang3.tuple.Pair;

import SVD.SVD;
import SVD.SVDTrainedModel;

import com.google.gson.Gson;
import commons.Utilities;

public class SimulatedAnnealing {
  
  private final Gson gson = new Gson();
  private final Random rng;
  
  private final List<WalmartQuery> queries;
  private final List<Pair<double[][], double[]>> results;
  
  private final double[] startingWeights;
  
  private final PriorityQueue<Weight> maxHeap;
  
  private final SVDTrainedModel userItem1Week;
  private final SVDTrainedModel userItem2Week;
  private final SVDTrainedModel userItem3Week;
  private final SVDTrainedModel userRank1Week;
  private final SVDTrainedModel userRank2Week;
  private final SVDTrainedModel userRank3Week;
  private final SVDTrainedModel itemRank1Week;
  private final SVDTrainedModel itemRank2Week;
  private final SVDTrainedModel itemRank3Week;
  
  private final SVDTrainedModel userFirstCategory;
  private final SVDTrainedModel userSecondCategory;
  
  private final Map<String, String[]> itemToCategories;
  
  public SimulatedAnnealing(double[] startingWeights, File queryFile, File directory, File itemToCategoriesFile) throws Exception{
    this.queries = new LinkedList<>();
    this.results = new LinkedList<>();
    this.rng = new Random();
    
    this.startingWeights = startingWeights;
    this.maxHeap = new PriorityQueue<>(5000, Collections.reverseOrder());
        
    userItem1Week = SVD.unserializeModel(new File(directory, "user_item_1week"));
    userItem2Week = SVD.unserializeModel(new File(directory, "user_item_2week"));
    userItem3Week = SVD.unserializeModel(new File(directory, "user_item_3week"));
    userRank1Week = SVD.unserializeModel(new File(directory, "user_rank_1week"));
    userRank2Week = SVD.unserializeModel(new File(directory, "user_rank_2week"));
    userRank3Week = SVD.unserializeModel(new File(directory, "user_rank_3week"));
    itemRank1Week = SVD.unserializeModel(new File(directory, "item_rank_1week"));
    itemRank2Week = SVD.unserializeModel(new File(directory, "item_rank_2week"));
    itemRank3Week = SVD.unserializeModel(new File(directory, "item_rank_3week"));
    userFirstCategory = SVD.unserializeModel(new File(directory, "user_item_cat1"));
    userSecondCategory = SVD.unserializeModel(new File(directory, "user_item_cat2"));
    System.out.println("SVD Models Unserialized");
    
    itemToCategories = Utilities.getItemToCategoryMapping(itemToCategoriesFile);
    
    getData(queryFile);
    System.out.println("Input Data Processed");
  }
  
  private double getScore(double[] weights){
    double score = 0;
    
    Iterator<WalmartQuery> queryIterator = queries.iterator();
    Iterator<Pair<double[][], double[]>> resultIterator = results.iterator();
    
    while (queryIterator.hasNext()){
      WalmartQuery query = queryIterator.next();
      Pair<double[][], double[]> result = resultIterator.next();
      score += getScore(query, result, weights);
    }
    
    return score;
  }
    
  /**
   * Hill climbing.
   */
  public void anneal(){
    
    double originalScore = getScore(this.startingWeights);
    System.out.println("Original Score: " + originalScore);
    
    double[] bestWeights = this.startingWeights;
    double bestScore = originalScore;

    maxHeap.add(new Weight(this.startingWeights, originalScore));
        
    while (! maxHeap.isEmpty()){
    
      Weight weight = maxHeap.remove();
      System.out.println("PQ Size: " + maxHeap.size());
      
      for (double[] perturbation : generatePerturbations(weight.weights.length)){
        
        double[] newWeights = new double[weight.weights.length];
        
        for (int i = 0; i < newWeights.length; i++){
          newWeights[i] = perturbation[i];
          if (perturbation[weight.weights.length] == 0){
            newWeights[i] += weight.weights[i];
          }
        }

        double score = getScore(newWeights);
        
        //System.out.println("Contender weights: " + Arrays.toString(newWeights));
        //System.out.println("Contender score: " + score);
        
        if ((score - bestScore) / bestScore > 0.002){
          maxHeap.add(new Weight(newWeights, score));
          
          if (score > bestScore){
            bestScore = score;
            bestWeights = newWeights;
            System.out.println("New Best Score: " + bestScore);
            System.out.println("New Best Weights: " + Arrays.toString(bestWeights));
          }
          
        }
      
      }
    
    }
    
    System.out.println("Best Score: " + bestScore);
    System.out.println("Best Weights: " + Arrays.toString(bestWeights));    
  }
  
  /**
   * Generate all weight shocks.
   */
  private List<double[]> generatePerturbations(int weightLength){
    
   double[] movements = new double[] { 0.5, 2.5 };
   
   List<double[]> perturbations = new LinkedList<>();
    
    for (int i = 0; i < weightLength; i++){
      for (int j = i; j < weightLength; j++){
        
        for (int k = 0; k < movements.length; k++){
        
          double[] firstPerturbation = new double[weightLength + 1];
          firstPerturbation[i] = movements[k] * Math.random();
          firstPerturbation[j] = movements[k] * Math.random();
          perturbations.add(firstPerturbation);
          
          double[] secondPerturbation = new double[weightLength + 1];
          secondPerturbation[i] = -movements[k] * Math.random();
          secondPerturbation[j] = -movements[k] * Math.random();
          perturbations.add(secondPerturbation);
          
          double[] fifthPerturbation = new double[weightLength + 1];
          fifthPerturbation[i] = -movements[k] * Math.random();
          fifthPerturbation[j] = -movements[k] * Math.random();
          fifthPerturbation[weightLength] = 1;
          perturbations.add(fifthPerturbation);
          
          double[] sixthPerturbation = new double[weightLength + 1];
          sixthPerturbation[i] = movements[k] * Math.random();
          sixthPerturbation[j] = movements[k] * Math.random();
          sixthPerturbation[weightLength] = 1;
          perturbations.add(sixthPerturbation);
          
          if (j > i){
            double[] thirdPerturbation = new double[weightLength + 1];
            thirdPerturbation[i] = movements[k] * Math.random(); thirdPerturbation[j] = -movements[k] * Math.random();
            perturbations.add(thirdPerturbation);
            
            double[] fourthPerturbation = new double[weightLength + 1];
            fourthPerturbation[i] = -movements[k] * Math.random(); fourthPerturbation[j] = movements[k] * Math.random();
            perturbations.add(fourthPerturbation);
            
            double[] seventhPerturbation = new double[weightLength + 1];
            seventhPerturbation[i] = movements[k] * Math.random(); seventhPerturbation[j] = -movements[k] * Math.random();
            seventhPerturbation[weightLength] = 1;
            perturbations.add(seventhPerturbation);
            
            double[] eigthPerturbation = new double[weightLength + 1];
            eigthPerturbation[i] = -movements[k] * Math.random(); eigthPerturbation[j] = movements[k] * Math.random();
            eigthPerturbation[weightLength] = 1;
            perturbations.add(eigthPerturbation);
            
          }
        }
      }
    }
    
    return perturbations;
    
  }
  
  public double getScore(WalmartQuery query, Pair<double[][], double[]> pair, double[] weights) {
    double[][] x = pair.getLeft();

    List<Pair<Double, Integer>> predictions = new ArrayList<>();
    for (int i = 0; i < x.length; ++i) {
      double[] xi = x[i];
      assert xi.length == weights.length;
      double guess = 0;
      for (int j = 0; j < xi.length; j++){
        guess += xi[j] * weights[j];
      }
      predictions.add(Pair.of(guess, i));
    }
    Collections.sort(predictions, Collections.reverseOrder());
    int[] ordering = new int[predictions.size()];
    for(int i = 0; i < ordering.length; ++i) {
      Pair<Double, Integer> prediction = predictions.get(i);
      ordering[i] = prediction.getRight();
    }
    return query.getPredictedScore(ordering);
  }
  
  public Pair<double[][], double[]> computeRegressionInput(WalmartQuery wquery) {
    String user = wquery.visitorid;
    int[] shownItems = wquery.shownitems;
    double[][] x = new double[shownItems.length][];
    for (int rank = 0; rank < shownItems.length; ++rank) {
      String item = "" + shownItems[rank];
      x[rank] = new double[11];
      x[rank][0] = userItem1Week.predict(user, item).getLeft();
      x[rank][1] = userItem2Week.predict(user, item).getLeft();
      x[rank][2] = userItem3Week.predict(user, item).getLeft();
      x[rank][3] = userRank1Week.predict(user, rank + "").getLeft();
      x[rank][4] = userRank2Week.predict(user, rank + "").getLeft();
      x[rank][5] = userRank3Week.predict(user, rank + "").getLeft();
      x[rank][6] = itemRank1Week.predict(item, rank + "").getLeft();
      x[rank][7] = itemRank2Week.predict(item, rank + "").getLeft();
      x[rank][8] = itemRank3Week.predict(item, rank + "").getLeft();
      if (itemToCategories.containsKey(item)){
        String[] categories = itemToCategories.get(item);
        x[rank][9] = userFirstCategory.predict(user, categories[0]).getLeft();
        x[rank][10] = userSecondCategory.predict(user, categories[1]).getLeft();
      }
    
    }
    
    double[] y = wquery.getRatings();
    return Pair.of(x, y);
  
  }
  
  public WalmartQuery parseQuery(String line) {
    try {
      return gson.fromJson(line, WalmartQuery.class);
    } catch(Exception e) {
      return null;
    }
  }
  
  /**
   * Read in Query data.
   * @param filename
   */
  private void getData(File filename) {
    
    double TO_KEEP = 0.4;
    try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
      String line = null;
      while ((line = reader.readLine()) != null) {
        WalmartQuery query = parseQuery(line);
        if (query == null) continue;
        if ((TO_KEEP < 1.0 && rng.nextDouble() <= TO_KEEP) || TO_KEEP == 1.0){
          queries.add(query);
          results.add(computeRegressionInput(query));
        }
        
      }
      
    } catch (IOException e){
      System.err.println("Error Reading in File");
      System.err.println(e);
      System.exit(1);
    }
    
  }
  
  public static void main(String[] args) throws Exception {
    
    long startTime = System.currentTimeMillis();
    
    Options options = new Options();
    options.addOption("tr", true, "Training File.");
    options.addOption("svd", true, "Directory containing serialized SVD models.");
    options.addOption("itc", true, "File Containing Item-To-Categories Mapping.");
    CommandLineParser parser = new GnuParser();
    CommandLine commandLine = parser.parse(options, args);
    
    double[] startingWeights = new double[]{0.5, 1.5, 3.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0, 1.0, 0};
    SimulatedAnnealing sa = new SimulatedAnnealing(startingWeights,
        new File(commandLine.getOptionValue("tr")), new File(commandLine.getOptionValue("svd")),
        new File(commandLine.getOptionValue("itc")));
    sa.anneal();
    
    long endTime = System.currentTimeMillis();
    System.out.println("Runtime: " + (((double) endTime - startTime) / 1000) + " seconds.");
  }

}

class Weight implements Comparable<Weight> {
  public final double[] weights;
  public final double score;
  
  public Weight(double[] weights, double score){
    this.weights = weights;
    this.score = score;
  }

  @Override
  public int compareTo(Weight o) {
    return Double.compare(this.score, o.score);
  }
  
}
