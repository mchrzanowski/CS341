package Regression;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.Options;
import org.apache.commons.lang3.tuple.Pair;
import org.paukov.combinatorics.Factory;
import org.paukov.combinatorics.Generator;
import org.paukov.combinatorics.ICombinatoricsVector;

import SVD.SVD;
import SVD.SVDTrainedModel;

import com.google.gson.Gson;
import commons.Utilities;

public class EnsembleWalmartSearchRank {

  private final Gson gson;
  private final SVDTrainedModel userItem1Week;
  private final SVDTrainedModel userItem2Week;
  private final SVDTrainedModel userItem3Week;
  private final SVDTrainedModel userItemCat1;
  private final SVDTrainedModel userItemCat2;
  private final SVDTrainedModel userRank1Week;
  private final SVDTrainedModel userRank2Week;
  private final SVDTrainedModel userRank3Week;
  private final SVDTrainedModel itemRank1Week;
  private final SVDTrainedModel itemRank2Week;
  private final SVDTrainedModel itemRank3Week;
  
  private final Map<String, String[]> itemToCategories;
  private final Map<Integer, String> itemToNames;
  private final Scheduler scheduler;
  private double[] weights;
  private List<double[]> perturbations;
  
  private Random rng;
  
  public EnsembleWalmartSearchRank(String directory, String itemToCategory, double[] weights) throws Exception {
    this.gson = new Gson();
    this.rng = new Random();
    this.scheduler = new Scheduler(10, 0.01, 100000);
    this.perturbations = generatePerturbations(weights.length);
    this.weights = weights;

    userItem1Week = SVD.unserializeModel(new File(directory, "user_item_1week"));
    userItem2Week = SVD.unserializeModel(new File(directory, "user_item_2week"));
    userItem3Week = SVD.unserializeModel(new File(directory, "user_item_3week"));
    userRank1Week = SVD.unserializeModel(new File(directory, "user_rank_1week"));
    userRank2Week = SVD.unserializeModel(new File(directory, "user_rank_2week"));
    userRank3Week = SVD.unserializeModel(new File(directory, "user_rank_3week"));
    itemRank1Week = SVD.unserializeModel(new File(directory, "item_rank_1week"));
    itemRank2Week = SVD.unserializeModel(new File(directory, "item_rank_2week"));
    itemRank3Week = SVD.unserializeModel(new File(directory, "item_rank_3week"));
    userItemCat1 = SVD.unserializeModel(new File(directory, "user_item_cat1"));
    userItemCat2 = SVD.unserializeModel(new File(directory, "user_item_cat2"));
    itemToCategories = Utilities.getItemToCategoryMapping(new File("/mnt/Archive/input_data/item_to_categories"));
    itemToNames = Utilities.getItemToNameyMapping(new File("/mnt/Archive/input_data/item_to_names"));
    System.out.println("Done loading SVD models...");
  }

  public void train(String filename) throws Exception {
    List<WalmartQuery> data = parseInput(filename);
    Node current = new Node(weights);
    //System.out.println("Current Weights: ");
    //System.out.println(getScores(data, new Node(new double[]{38.5, 47.5, 53.0, 56.5, 36.5, 37.5, 11.5, 9.0, 16.0, 12.0, 20.0, 36.5})));
    //System.exit(0);
    Node next = null;
    int timeStep = 0;
    while (true) {
      double temperature = scheduler.getTemp(timeStep++);
      if (temperature == 0.0) break;
      List<Node> children = expandNode(current);
      if(!children.isEmpty()) {
        int pick = (int)(rng.nextDouble() * children.size());
        next = children.get(pick);
        double deltaE = getScores(data, next) - getScores(data, current);
        if (shouldAccept(temperature, deltaE)) {
          System.out.println(next.toString());
          current = next;
        }
      }
    }
  }

  public void test(String filename) throws Exception {
    BufferedWriter writer = new BufferedWriter(new FileWriter("error.histogram"));
    List<WalmartQuery> data = parseInput(filename);
    double baselineScore = 0;
    double testScore = 0;
    int queries = data.size();
    double bestScore = 0;
    for(WalmartQuery query : data) {
      double baseline = query.getBaselineScore();
      double score    = getScore(query, weights);
      writer.write(String.format("%f\n", score - baseline));
      baselineScore += baseline;
      testScore += score;
      bestScore += query.getMaxScore();
    }
    writer.close();
    System.out.println("Queries: " + queries);
    System.out.printf("Baseline Score: %f\n", baselineScore);
    System.out.printf("Test Score: %f\n", testScore);
    System.out.printf("Precent Improvement: %f\n",
        100 * (testScore - baselineScore) / baselineScore);
    System.out.println("Score: " + (testScore - baselineScore));
    System.out.println("Best Score: " + bestScore);
  }

  /////////////////////////////////////////////////////////////////////////////
  // HELPER METHODS
  /////////////////////////////////////////////////////////////////////////////

  public List<WalmartQuery> parseInput(String filename) throws Exception {
    try (BufferedReader reader = new BufferedReader(new FileReader(new File(filename)))) {
      String line = null;
      List<WalmartQuery> queries = new LinkedList<>();
      while ((line = reader.readLine()) != null) {
        WalmartQuery query = parseQuery(line);
        if (query == null) continue;
        query.svdinputs = computeSVDInput(query);
        queries.add(query);
      }
      return queries;
    }
  }

  public double[][] computeSVDInput(WalmartQuery wquery) {
    String user = wquery.visitorid;
    int[] shownItems = wquery.shownitems;
    double[][] x = new double[shownItems.length][];
    for (int rank = 0; rank < shownItems.length; ++rank) {
      String item = "" + shownItems[rank];
      x[rank] = new double[11];
      x[rank][0] = userItem1Week.predict(user, item).getLeft();
      x[rank][1] = userItem2Week.predict(user, item).getLeft();
      x[rank][2] = userItem3Week.predict(user, item).getLeft();      
      if (this.itemToCategories.containsKey(item)){
        String[] categories = itemToCategories.get(item);
        x[rank][3] = userItemCat1.predict(user, categories[0]).getLeft();
        x[rank][4] = userItemCat2.predict(user, categories[1]).getLeft();
      }
      x[rank][5] = userRank1Week.predict(user, rank + "").getLeft();
      x[rank][6] = userRank2Week.predict(user, rank + "").getLeft();
      x[rank][7] = userRank3Week.predict(user, rank + "").getLeft();
      x[rank][8] = itemRank1Week.predict(item, rank + "").getLeft();
      x[rank][9] = itemRank2Week.predict(item, rank + "").getLeft();
      x[rank][10] = itemRank3Week.predict(item, rank + "").getLeft();
    }
    return x;
  }

  public WalmartQuery parseQuery(String line) {
    try {
      WalmartQuery query = gson.fromJson(line, WalmartQuery.class);
      return query.clickeditems.length > 0 ? query : null;
    } catch(Exception e) {
      return null;
    }
  }
    
  /**
   * Create perturbations. Use a rather fat-tailed Gaussian RNG to reduce the need to store many discrete perturbations.
   * @param indices
   * @param perturbations
   * @param movements
   * @param index
   * @param current
   * @param weightLength
   */
  private void generatePerturbations(List<Integer> indices, List<double[]> perturbations, double[] movements, int index, double[] current, int weightLength){
    
    if (index >= indices.size() && current != null){
      perturbations.add(current);
      return;
    }
    
    for (int i = 0; i < movements.length; i++){
      
      double[] newArray = new double[weightLength];
      if (index != 0){
        newArray = Arrays.copyOf(current, current.length);
      }
      // mean = movements[i], stdev = 2
      newArray[indices.get(index)] = this.rng.nextGaussian() * 2 + movements[i];
      generatePerturbations(indices, perturbations, movements, index + 1, newArray, weightLength);
    }
        
  }
  
  private List<double[]> generatePerturbations(int weightLength){
    double[] movements = new double[] {-0.75, 0.75};
    List<double[]> perturbations = new ArrayList<>();
    
    Integer[] values = new Integer[weightLength];
    for (int i = 0; i < values.length; i++){
      values[i] = i;
    }
    
    ICombinatoricsVector<Integer> initialSet = Factory.createVector(values);
    Generator<Integer> generator = Factory.createSubSetGenerator(initialSet);
    
    for (ICombinatoricsVector<Integer> subset : generator) {
      List<Integer> indices = subset.getVector();
      if (indices.size() == 0) continue;
      List<double[]> perturbs = new ArrayList<>();
      generatePerturbations(indices, perturbs, movements, 0, null, weightLength);
      perturbations.addAll(perturbs);
      
    }
    
    System.out.println(perturbations.size());
      
    return perturbations;
  }

  public List<Node> expandNode(Node node) {  
    List<Node> children = new ArrayList<>();
    for (double[] perturbation : perturbations){
      double[] newWeights = new double[node.weights.length];
      for (int i = 0; i < newWeights.length; i++){
        newWeights[i] = node.weights[i] + perturbation[i];
      }
      children.add(new Node(newWeights));
    }
    return children;
  }

  public boolean shouldAccept(double temperature, double deltaE) {
    return (deltaE > 0.0) || 
        (rng.nextDouble() <= Math.exp(deltaE / temperature));
  }

  public double predict(double[] x, double[] w) {
    double total = w[0];
    for (int i = 0; i < x.length; ++i) {
      total += x[i] * w[i + 1];
    }
    return total;
  }

  public double getScores(List<WalmartQuery> data, Node node) {
    if(!Double.isNaN(node.score)) return node.score;     
    double total = 0;
    for(WalmartQuery query : data) {
      total += getScore(query, node.weights);
    }
    node.score = total;
    return total;
  }

  public double getScore(WalmartQuery query, double[] w) {
    double[][] x = query.svdinputs;
    List<Pair<Double, Integer>> predictions = new ArrayList<>();
    for (int i = 0; i < x.length; ++i) {
      double[] xi = x[i];
      double guess = predict(xi, w);
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

  class Node implements Comparable<Node> {
    public final double[] weights;
    public double score;
    
    public Node(double[] weights){
      this.weights = weights;
      this.score = Double.NaN;
    }

    @Override
    public int compareTo(Node o) {
      return Double.compare(this.score, o.score);
    }    

    public String toString() {
      return "score=" + this.score + 
          " weights=" + Arrays.toString(this.weights);
    }
  }

  public static void runTest(EnsembleWalmartSearchRank wsr, double[] weights) throws Exception {
    wsr.weights = weights;
    System.out.println("Weights:\n" + Arrays.toString(wsr.weights));
    System.out.println("Testing: ");
    long startTime = System.currentTimeMillis();
    wsr.test("input_data/query/testing/good_testing");
    long endTime = System.currentTimeMillis();
    double duration = (endTime - startTime) / 1000;
    System.out.println("Runtime: " + duration + " seconds.");
    System.out.println("######################################################################");
  }

  public static void main(String[] args) throws Exception {    
    Options options = new Options();
    options.addOption("tr", true, "Training File.");
    options.addOption("te", true, "Testing File.");
    options.addOption("svd", true, "Directory containing serialized SVD models.");
    options.addOption("itc", true, "File Containing Item-To-Categories Mapping.");
    CommandLineParser parser = new GnuParser();
    CommandLine commandLine = parser.parse(options, args);
    
    EnsembleWalmartSearchRank wsr = new EnsembleWalmartSearchRank(commandLine.getOptionValue("svd"),
        commandLine.getOptionValue("itc"),
        new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    wsr.train(commandLine.getOptionValue("tr"));
    // wsr.test(commandLine.getOptionValue("te"));

    /*
    // wsr.train("input_data/query/testing/good_testing");
    // wsr.test("input_data/query/testing/good_testing");
    runTest(wsr, new double[] { 0.0, 0.5, 1.5, 3.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0});
    runTest(wsr, new double[] { 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0});
    runTest(wsr, new double[] { 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0});
    runTest(wsr, new double[] { 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 1.0});
    runTest(wsr, new double[] { 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.5});
    runTest(wsr, new double[] { 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 1.0});
    runTest(wsr, new double[] { 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 1.5});

    runTest(wsr, new double[] { 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    runTest(wsr, new double[] { 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    runTest(wsr, new double[] { 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    runTest(wsr, new double[] { 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    runTest(wsr, new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    runTest(wsr, new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    runTest(wsr, new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0});
    runTest(wsr, new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0});
    runTest(wsr, new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0});
    runTest(wsr, new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0});
    runTest(wsr, new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0});
    */
  }
  

}
