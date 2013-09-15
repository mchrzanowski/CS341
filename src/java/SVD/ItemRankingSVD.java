package SVD;

import java.io.File;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import commons.FileParsing;
import commons.Triplet;


public class ItemRankingSVD extends SVD {
	
	public ItemRankingSVD(int concepts, int maxIterations, List<Triplet> inputData,
			double lambda, boolean verbose, Map<String, Integer> firstArgumentTranslation,
			Map<String, Integer> secondArgumentTranslation,
			Set<Integer> labels){
		
		super(concepts, maxIterations, inputData, lambda, verbose,
				firstArgumentTranslation, secondArgumentTranslation, labels);
		
		initializeLearningRates();
		
		if (verbose){
			printInitialParameters();
		}
	}
	
	public ItemRankingSVD(int concepts, int maxIterations, File trainingFile,
			double lambda, boolean verbose){
		
		super(concepts, maxIterations, lambda, verbose, trainingFile);
		
		initializeLearningRates();

		if (verbose){
			printInitialParameters();
		}
	}
	
	private void printInitialParameters(){
		System.out.println("Parameters:");
		System.out.printf("Concepts:\t%d\n", getConcepts());
		System.out.printf("Iterations:\t%d\n", getMaxIterations());
		System.out.printf("Lambda:\t\t%.4f\n", getTrainedModel().getLambda());
		System.out.println("Etas:\t\t" + this.getEtas());
		System.out.println();
		System.out.println("Data:");
		System.out.printf("Items:\t\t%d\n", getTrainedModel().getFirstArgumentTranslation().size());
		System.out.printf("Rankings:\t%d\n", getTrainedModel().getSecondArgumentTranslation().size());
		System.out.printf("Labels:\t\t%d\n", getTrainedModel().getLabels().size());
	}
	
	private void initializeLearningRates(){
		Map<Integer, Double> etas = new HashMap<>();
		etas.put(1, 0.03);
		etas.put(2, 0.03);
		etas.put(3, 0.03);
		etas.put(4, 0.03);
		
		this.setEtas(etas);
	}
		
	
	/**
	 * Perform K-Fold cross validation.
	 */
	public static void crossValidate(int concepts, int maxIterations, File trainingFile,
			double lambda, boolean verbose) {
		
		Map<String, Integer> firstTranslation = new HashMap<>();
		Map<Integer, String> reverseFirstTranslation = new HashMap<>();
		Map<String, Integer> secondTranslation = new HashMap<>();
		Map<Integer, String> reverseSecondTranslation = new HashMap<>();
		Set<Integer> labels = new HashSet<>();
		
		List<Triplet> inputData = FileParsing.parseInputFile(trainingFile, 
				firstTranslation, reverseFirstTranslation, secondTranslation, reverseSecondTranslation,
				labels, true, true);
				
		int KFOLD = SVD.getK_FOLD();
		
		for (int i = 0; i < KFOLD; i++){
			System.out.printf("Cross Validation Test %d of %d\n", i + 1, KFOLD);
			// partition data.
			int pivot = i * (inputData.size() / SVD.getK_FOLD());
			int nextPivot = (int) Math.min(inputData.size(), (i + 1) * Math.ceil(((double) inputData.size()) / KFOLD));

			List<Triplet> trainingData = new LinkedList<>(inputData.subList(0, pivot));
			trainingData.addAll(inputData.subList(nextPivot, inputData.size()));
			
			Map<String, Integer> cvFirstTranslation = new HashMap<>();
			Map<String, Integer> cvSecondTranslation = new HashMap<>();
			
			Set<Integer> seenFirstArgument = new HashSet<>();
			Set<Integer> seenSecondArgument = new HashSet<>();
						
			for (Triplet triplet : trainingData){
				int first = triplet.first;
				int second = triplet.second;
				
				if (! seenFirstArgument.contains(first)){
					cvFirstTranslation.put(reverseFirstTranslation.get(first), first);
					seenFirstArgument.add(first);
				}
				
				if (! seenSecondArgument.contains(second)){
					cvSecondTranslation.put(reverseSecondTranslation.get(second), second);
					seenSecondArgument.add(second);
				}	
			}
			
			List<Triplet> testingData = new LinkedList<>(inputData.subList(pivot, nextPivot));
			Iterator<Triplet> testingIterator = testingData.iterator();
			while (testingIterator.hasNext()){
				Triplet triplet = testingIterator.next();
				if (! seenFirstArgument.contains(triplet.first) || ! seenSecondArgument.contains(triplet.second)){
					testingIterator.remove();
				}
			}

			SVD svd = new ItemRankingSVD(concepts, maxIterations, trainingData, lambda, verbose,
					cvFirstTranslation, cvSecondTranslation, labels);
			svd.train();
			
			svd.test(testingData);
			
		}
	}

	/**
	 * @param args
	 * @throws ParseException 
	 */
	public static void main(String[] args) throws ParseException {
		long startTime = System.currentTimeMillis();
		
		Options options = new Options();
		options.addOption("cv", false, "Perform cross validation.");
		options.addOption("k", true, "Concepts");
		options.addOption("n", true, "Iteration number");
		options.addOption("l", true, "Lambda Regularization Parameter");
		options.addOption("s", true, "Serialize model after training.");
		options.addOption("tr", true, "Training File.");
		options.addOption("te", true, "Testing File.");
		options.addOption("v", false, "Verbose mode.");

		CommandLineParser parser = new GnuParser();
		CommandLine commandLine = parser.parse(options, args);
		
		int concepts = 30;
		if (commandLine.hasOption("k")){
			concepts = Integer.parseInt(commandLine.getOptionValue("k"));
		}
		
		int iterations = 40;
		if (commandLine.hasOption("n")){
			iterations = Integer.parseInt(commandLine.getOptionValue("n"));
		}
		
		double lambda = 0.2;
		if (commandLine.hasOption("l")){
			lambda = Double.parseDouble(commandLine.getOptionValue("l"));
		}
		
		boolean verbose = false;
		if (commandLine.hasOption("v")){
			verbose = true;
		}
		
		boolean crossValidate = false;
		if (commandLine.hasOption("cv")){
			crossValidate = true;
		}
		
		if (crossValidate){
			ItemRankingSVD.crossValidate(concepts, iterations,
					new File(commandLine.getOptionValue("tr")), lambda, verbose);
		}
		else {
			ItemRankingSVD svd = new ItemRankingSVD(concepts, iterations,
					new File(commandLine.getOptionValue("tr")), lambda, verbose);
			svd.train();
			
			System.out.println("Training Errors:");
			svd.test(new File(commandLine.getOptionValue("tr")));
			
			System.out.println("Testing Errors:");
			svd.test(new File(commandLine.getOptionValue("te")));
			
			if (commandLine.hasOption("s")){
				SVD.serializeModel(svd, new File(commandLine.getOptionValue("s")));
			}
		}
		
		long endTime = System.currentTimeMillis();
		System.out.println("Runtime: " + (((double) endTime - startTime) / 1000) + " seconds.");
	}
}

