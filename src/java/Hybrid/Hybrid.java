package Hybrid;

import java.io.File;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import commons.FileParsing;
import commons.Metrics;
import commons.Prediction;

import RBM.Pair;
import RBM.RBM;
import SVD.UserItemSVD;

public class Hybrid {
	
	public static NavigableMap<Integer, List<Pair>> input;
	
	public static List<Prediction> svdTrainingPredictions;
	public static List<Prediction> svdTestingPredictions;
	
	public static List<Prediction> rbmTrainingPredictions;
	public static List<Prediction> rbmTestingPredictions;
	
	
	public static void getRBMPredictions(int batchSize, int features, int iterations, File trainingFile, 
			File testingFile, boolean verbose){
		
		RBM rbm = new RBM(batchSize, features, verbose, trainingFile, iterations);
		rbm.train();
		
		NavigableMap<Integer, List<Pair>> testingData = FileParsing.parseUserItemFile(testingFile,
				rbm.getItemTranslation(), rbm.getUserTranslation(), rbm.getLabels(), false, true);
		
		NavigableMap<Integer, List<Pair>> trainingData = FileParsing.parseUserItemFile(trainingFile,
				rbm.getItemTranslation(), rbm.getUserTranslation(), rbm.getLabels(), false, true);
		
		rbmTestingPredictions = rbm.predict(testingData);
		Metrics.collapse(rbmTestingPredictions, rbm.getLabels(), rbm.isVerbose());
		
		rbmTrainingPredictions = rbm.predict(trainingData);
		Metrics.collapse(rbmTrainingPredictions, rbm.getLabels(), rbm.isVerbose());
	
	}
	

	public static void getSVDPredictions(int concepts, int iterations, File trainingFile, 
			File testingFile, double lambda, boolean verbose){
		
		Map<Integer, Double> etas = new HashMap<>();
		etas.put(1, 0.001);
		etas.put(2, 0.06);
		etas.put(3, 0.08);
		etas.put(4, 0.1);
		
		UserItemSVD svd = new UserItemSVD(concepts, iterations, trainingFile, lambda, verbose, etas);
		svd.train();
		
		NavigableMap<Integer, List<Pair>> testingData = FileParsing.parseUserItemFile(testingFile,
				svd.getItemTranslation(), svd.getUserTranslation(), svd.getLabels(), false, false);
		
		NavigableMap<Integer, List<Pair>> trainingData = FileParsing.parseUserItemFile(trainingFile,
				svd.getItemTranslation(), svd.getUserTranslation(), svd.getLabels(), false, false);
		
		svdTestingPredictions = svd.predict(testingData);
		Metrics.collapse(svdTestingPredictions, svd.getLabels(), svd.isVerbose());
		
		svdTrainingPredictions = svd.predict(trainingData);
		Metrics.collapse(svdTrainingPredictions, svd.getLabels(), svd.isVerbose());
		
		input = svd.getInputData();
		
	}
	
	public static List<Prediction> consolidate(List<Prediction> svdPredictions, List<Prediction> rbmPredictions){
		Iterator<Prediction> svdIterator = svdPredictions.iterator();
		Iterator<Prediction> rbmIterator = rbmPredictions.iterator();
		
		while (svdIterator.hasNext()){
			Prediction sPrediction = svdIterator.next();
			Prediction rPrediction = rbmIterator.next();
			
			if (sPrediction.pair.score != (rPrediction.pair.score + 1) ||
					sPrediction.pair.item != rPrediction.pair.item){
				System.err.println("ERROR");
				System.out.println(sPrediction.pair.toString());
				System.out.println(rPrediction.pair.toString());
				System.exit(1);
			}
			
			if (rPrediction.prediction == 0){
				sPrediction.prediction = 1;
			}
			
		}
		
		return svdPredictions;
	}

	/**
	 * @param args
	 * @throws ParseException 
	 */
	public static void main(String[] args) throws ParseException {
		long startTime = System.currentTimeMillis();
		
		Options options = new Options();
		options.addOption("k", true, "Concepts");
		options.addOption("sn", true, "SVD epochs");
		options.addOption("rn", true, "RBM epochs");
		options.addOption("l", true, "Lambda Regularization Parameter");
		options.addOption("tr", true, "Training File.");
		options.addOption("te", true, "Testing File.");
		options.addOption("v", false, "Verbose mode.");
		options.addOption("b", true, "Batch size.");
		options.addOption("f", true, "Features");

		CommandLineParser parser = new GnuParser();
		CommandLine commandLine = parser.parse(options, args);
		
		int concepts = 30;
		if (commandLine.hasOption("k")){
			concepts = Integer.parseInt(commandLine.getOptionValue("k"));
		}
		
		int svdIterations = 40;
		if (commandLine.hasOption("sn")){
			svdIterations = Integer.parseInt(commandLine.getOptionValue("sn"));
		}
		
		int rbmIterations = 40;
		if (commandLine.hasOption("rn")){
			rbmIterations = Integer.parseInt(commandLine.getOptionValue("rn"));
		}
		
		double lambda = 0.2;
		if (commandLine.hasOption("l")){
			lambda = Double.parseDouble(commandLine.getOptionValue("l"));
		}
		
		boolean verbose = false;
		if (commandLine.hasOption("v")){
			verbose = true;
		}
		
		int batchSize = 100;
		if (commandLine.hasOption("b")) {
			batchSize = Integer.parseInt(commandLine.getOptionValue("b"));
		}

		int features = 50;
		if (commandLine.hasOption("f")) {
			features = Integer.parseInt(commandLine.getOptionValue("f"));
		}
		
		File trainingFile = new File(commandLine.getOptionValue("tr"));
		File testingFile = new File(commandLine.getOptionValue("te"));

		getSVDPredictions(concepts, svdIterations, trainingFile, testingFile, lambda, verbose);
		getRBMPredictions(batchSize, features, rbmIterations, trainingFile, testingFile, verbose);
		
		List<Prediction> trainingConsolidated = consolidate(svdTrainingPredictions, rbmTrainingPredictions);
		List<Prediction> testingConsolidated = consolidate(svdTestingPredictions, rbmTestingPredictions);

		Map<Integer, Double> weights = Metrics.getWeightsOfLabels(input);
		
		System.out.println("Training Error:");
		Metrics.evaluatePrecisionAndRecall(trainingConsolidated);
		Metrics.evaluatePointsScored(trainingConsolidated, weights);
		
		System.out.println("Testing Error:");
		Metrics.evaluatePrecisionAndRecall(testingConsolidated);
		Metrics.evaluatePointsScored(testingConsolidated, weights);
		
		
		long endTime = System.currentTimeMillis();
		System.out.println("Runtime: " + (((double) endTime - startTime) / 1000) + " seconds.");

	}
	
}
