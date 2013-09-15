package Regression;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
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

public class NewRegression {
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
  
  private Random rng;
  
  private final double eta;
  private final int iterations;

    
  public NewRegression(File directory, File itemToCategory, File itemToName, double eta, int iterations) throws Exception {
    this.gson = new Gson();
    this.rng = new Random();
    this.eta = eta;
    this.iterations = iterations;

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
    System.out.println("Done loading SVD models...");

    itemToCategories = Utilities.getItemToCategoryMapping(itemToCategory);
    itemToNames = Utilities.getItemToNameyMapping(itemToName);
  }
  
  public double[][] computeSVDInput(WalmartQuery wquery) {
    String user = wquery.visitorid;
    int[] shownItems = wquery.shownitems;
    double[][] x = new double[shownItems.length][];
    for (int rank = 0; rank < shownItems.length; ++rank) {
      String item = "" + shownItems[rank];
      x[rank] = new double[12];
      x[rank][0] = 1;
      x[rank][1] = userItem1Week.predict(user, item).getLeft();
      x[rank][2] = userItem2Week.predict(user, item).getLeft();
      x[rank][3] = userItem3Week.predict(user, item).getLeft();      
      if (this.itemToCategories.containsKey(item)){
        String[] categories = itemToCategories.get(item);
        x[rank][4] = userItemCat1.predict(user, categories[0]).getLeft();
        x[rank][5] = userItemCat2.predict(user, categories[1]).getLeft();
      }
      x[rank][6] = userRank1Week.predict(user, rank + "").getLeft();
      x[rank][7] = userRank2Week.predict(user, rank + "").getLeft();
      x[rank][8] = userRank3Week.predict(user, rank + "").getLeft();
      x[rank][9] = itemRank1Week.predict(item, rank + "").getLeft();
      x[rank][10] = itemRank2Week.predict(item, rank + "").getLeft();
      x[rank][11] = itemRank3Week.predict(item, rank + "").getLeft();
    }
    return x;
  }
  
  public double[] train(List<WalmartQuery> training) throws Exception {
    
    Collections.shuffle(training);
    double[] weights = new double[12];
    for (int i = 0; i < weights.length; i++){
      weights[i] = rng.nextGaussian(); 
    }
    
    System.out.println("Beginning Training.");
    for (int iter = 0; iter < iterations; iter++){
      System.out.println("Iteration: " + (iter + 1));
      for (WalmartQuery query : training){
        
        int[] predictedY = getOrdering(query, weights);
        int[] bestY = getOptimalOrdering(query);
        
        assert bestY.length == predictedY.length;
        assert predictedY.length == query.shownitems.length;
        
        for (int i = 0; i < bestY.length; i++){
          double error = -bestY[i] + predictedY[i];
          for (int j = 0; j < query.svdinputs[i].length; j++){
            weights[j] += eta * error * query.svdinputs[i][j];
          }
        }
      }
     
     System.out.println(Arrays.toString(weights));
     System.out.println("Training Error");
     test(training, weights);
     
    }
    
    return weights;
  }
  
  public double getScore(WalmartQuery query, double[] w) {
    double[][] x = query.svdinputs;
    List<Pair<Double, Integer>> predictions = new ArrayList<>();
    for (int i = 3; i < x.length; ++i) {
      double[] xi = x[i];
      double guess = predict(xi, w);
      predictions.add(Pair.of(guess, i));
    }
    Collections.sort(predictions, Collections.reverseOrder());
    int[] ordering = new int[x.length];
    for (int i = 0; i < 3 && i < x.length; i++){
      ordering[i] = i;
    }
    for(int i = 0; i < predictions.size(); ++i) {
      Pair<Double, Integer> prediction = predictions.get(i);
      ordering[i+3] = prediction.getRight();
    }
    return query.getPredictedScore(ordering);
  }
  
  public void testAndDisplay(List<WalmartQuery> data, double[] weights) throws Exception {
    
    BufferedWriter writer = new BufferedWriter(new FileWriter("error.histogram"));
    
    // max heap
    PriorityQueue<Pair<Double, WalmartQuery>> maxPQ = new PriorityQueue<>(data.size(), Collections.reverseOrder());
    PriorityQueue<Pair<Double, WalmartQuery>> minPQ = new PriorityQueue<>(data.size());

    double baselineScore = 0;
    double testScore = 0;
    int queries = data.size();
    double bestScore = 0;
    for(WalmartQuery query : data) {
      double baseline = query.getBaselineScore();
      double score    = getScore(query, weights);
      baselineScore += baseline;
      testScore += score;
      bestScore += query.getMaxScore();
      maxPQ.add(Pair.of(score - baseline, query));
      minPQ.add(Pair.of(score - baseline, query));
    }
    
    writer.write("Good Ones:\n");
    for (int i = 0; i < 10; i++){
      WalmartQuery query = maxPQ.remove().getRight();
      int[] ourOrdering = getOrdering(query, weights);
      int[] mappedOrdering = new int[ourOrdering.length];
      for (int j = 0; j < ourOrdering.length; j++){
        mappedOrdering[j] = query.shownitems[ourOrdering[j]];
      }
      
      double score = getScore(query, weights);
      double theirScore = query.getBaselineScore();
      writer.write("Query: " + query.rawquery);
      writer.write("Our Score: " + score + "\n");
      writer.write("Their Score: " + theirScore + "\n");
      writer.write("Our Ordering:\n");
      writer.write(WalmartQuery.getItemStrings(query, mappedOrdering, itemToNames));
      writer.write("Their Ordering:\n");
      writer.write(WalmartQuery.getItemStrings(query, query.shownitems, itemToNames)); 
    }
    
    writer.write("\nBad ones: \n");
    for (int i = 0; i < 10; i++){
      WalmartQuery query = minPQ.remove().getRight();
      int[] ourOrdering = getOrdering(query, weights);
      int[] mappedOrdering = new int[ourOrdering.length];
      for (int j = 0; j < ourOrdering.length; j++){
        mappedOrdering[j] = query.shownitems[ourOrdering[j]];
      }
      double score = getScore(query, weights);
      double theirScore = query.getBaselineScore();
      writer.write("Query: " + query.rawquery);
      writer.write("Our Score: " + score + "\n");
      writer.write("Their Score: " + theirScore + "\n");
      writer.write("Our Ordering:\n");
      writer.write(WalmartQuery.getItemStrings(query, mappedOrdering, itemToNames));
      writer.write("Their Ordering:\n");
      writer.write(WalmartQuery.getItemStrings(query, query.shownitems, itemToNames)); 
    }
    
    writer.close();
   
    System.out.println("Queries: " + queries);
    System.out.printf("Baseline Score: %f\n", baselineScore);
    System.out.printf("Reranked Score: %f\n", testScore);
    System.out.printf("Percent Improvement: %f\n",
        100 * (testScore - baselineScore) / baselineScore);
    System.out.println("Score: " + (testScore - baselineScore));
    System.out.println("Best Score: " + bestScore);
  }
  
  public void test(List<WalmartQuery> data, double[] weights) throws Exception {
    
    double baselineScore = 0;
    double testScore = 0;
    int queries = data.size();
    double bestScore = 0;
    for(WalmartQuery query : data) {
      double baseline = query.getBaselineScore();
      double score    = getScore(query, weights);
      baselineScore += baseline;
      testScore += score;
      bestScore += query.getMaxScore();
    }
   
    System.out.println("Queries: " + queries);
    System.out.printf("Baseline Score: %f\n", baselineScore);
    System.out.printf("Reranked Score: %f\n", testScore);
    System.out.printf("Percent Improvement: %f\n",
        100 * (testScore - baselineScore) / baselineScore);
    System.out.println("Score: " + (testScore - baselineScore));
    System.out.println("Best Score: " + bestScore);
  }
  
  public void test(File dataFile, double[] weights) throws Exception {
    
    double baselineScore = 0;
    double testScore = 0;
    int queries = 0;
    double bestScore = 0;
    try (BufferedReader reader = new BufferedReader(new FileReader(dataFile))) {
      String line = null;
      while ((line = reader.readLine()) != null) {
        WalmartQuery query = parseQuery(line);
        if (query == null) continue;
        query.svdinputs = computeSVDInput(query);
        queries++;          
        double baseline = query.getBaselineScore();
        double score    = getScore(query, weights);
        baselineScore += baseline;
        testScore += score;
        bestScore += query.getMaxScore();
      }
    }
   
    System.out.println("Queries: " + queries);
    System.out.printf("Baseline Score: %f\n", baselineScore);
    System.out.printf("Reranked Score: %f\n", testScore);
    System.out.printf("Percent Improvement: %f\n", 100 * (testScore - baselineScore) / baselineScore);
    System.out.println("Score: " + (testScore - baselineScore));
    System.out.println("Best Score: " + bestScore);
  }
  
  public double predict(double[] x, double[] w) {
    double total = 0;
    for (int i = 0; i < x.length; ++i) {
      total += x[i] * w[i];
    }
    return total;
  }
  
  public int[] getOrdering(WalmartQuery query, double[] w) {
    double[][] x = query.svdinputs;
    List<Pair<Double, Integer>> predictions = new ArrayList<>();
    for (int i = 3; i < x.length; ++i) {
      double[] xi = x[i];
      double guess = predict(xi, w);
      predictions.add(Pair.of(guess, i));
    }
    Collections.sort(predictions, Collections.reverseOrder());
    int[] ordering = new int[x.length];
    
    for (int i =0; i < 3 && i < x.length; i++){
      ordering[i] = i;
    }
    
    for(int i = 3; i < x.length; ++i) {
      
      for (int k = 0; k < predictions.size(); k++){
        if (predictions.get(k).getRight() == i){
          ordering[i] = k + 3;
          break;
        }
      }
      
    }
    return ordering;
  }
  
  public int[] getOptimalOrdering(WalmartQuery query){
    int[] ordering = new int[query.shownitems.length];
    for (int i = 0; i < ordering.length; i++){
      ordering[i] = -1;
    }
    int currentIndex = 0;
    
    for (int i = 0; i < query.shownitems.length; ++i) {
      if (query.find(query.clickeditems, query.shownitems[i]) >= 0){
        ordering[i] = currentIndex++;
      }
    }
    
    for (int i = 0; i < query.shownitems.length; ++i) {
      if (ordering[i] == -1){
        ordering[i] = currentIndex++;
      }
    }
    
    return ordering;
  }
  
  public WalmartQuery parseQuery(String line) {
    try {
      WalmartQuery query = gson.fromJson(line, WalmartQuery.class);
      return query.clickeditems.length > 0 ? query : null;
    } catch(Exception e) {
      return null;
    }
  }
  
  public static List<WalmartQuery> parseInput(NewRegression nr, File filename) throws Exception {
    try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
      String line = null;
      List<WalmartQuery> queries = new ArrayList<>();
      while ((line = reader.readLine()) != null) {
        WalmartQuery query = nr.parseQuery(line);
        if (query == null) continue;
        query.svdinputs = nr.computeSVDInput(query);
        queries.add(query);
      }
      return queries;
    }
  }
  
  public static void main(String[] args) throws Exception {
    long startTime = System.currentTimeMillis();
    Options options = new Options();
    options.addOption("tr", true, "Training File.");
    options.addOption("te", true, "Testing File.");
    options.addOption("svd", true, "Directory containing serialized SVD models.");
    options.addOption("itc", true, "File Containing Item-To-Categories Mapping.");
    options.addOption("itn", true, "File Containing Item-To-Name Mapping.");
    options.addOption("eta", true, "File Containing Item-To-Categories Mapping.");
    options.addOption("n", true, "File Containing Item-To-Categories Mapping.");

    CommandLineParser parser = new GnuParser();
    CommandLine commandLine = parser.parse(options, args);
    
    int iterations = 1;
    if (commandLine.hasOption("n")){
      iterations = Integer.parseInt(commandLine.getOptionValue("n"));
    }
    
    double eta = 0.01;
    if (commandLine.hasOption("eta")){
      eta = Double.parseDouble(commandLine.getOptionValue("eta"));
    }
    
    NewRegression lr = new NewRegression(new File(commandLine.getOptionValue("svd")),
        new File(commandLine.getOptionValue("itc")), new File(commandLine.getOptionValue("itn")), eta, iterations);
    
    double[] weights = null;
    
    if (commandLine.hasOption("tr")){
      List<WalmartQuery> trainingData = NewRegression.parseInput(lr, new File(commandLine.getOptionValue("tr")));
      weights = lr.train(trainingData);
      System.out.println("Training Error...");
      lr.test(trainingData, weights);
      
      trainingData.clear();
    }
    
    else {
    
      weights = new double[] {-1.183934084095191, 0.03863952261853375,
          0.05388661872794404, 0.026366184120903045, 0.03208445037792477,
          0.01419800031390554, 0.10351658717560684, -0.009427816710883656,
          -0.009722266282593414, -0.019403715926362274, 0.05366842947244714,
          0.05682652702849528};
    }
    System.out.println("Weights:\n" + Arrays.toString(weights));
    
    if (commandLine.hasOption("te")){
      System.out.println("Testing Error...");
      lr.test(new File(commandLine.getOptionValue("te")), weights);
    }
    
    long endTime = System.currentTimeMillis();
    double duration = (endTime - startTime) / 60000d;
    System.out.printf("Runtime: %f minutes.\n", duration);


  }

}