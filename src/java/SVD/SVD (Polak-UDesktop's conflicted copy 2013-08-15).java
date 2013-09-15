package SVD;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math.linear.BlockRealMatrix;
import org.apache.commons.math.linear.RealMatrix;
import org.apache.commons.math.linear.RealVector;

import commons.FileParsing;
import commons.Metrics;
import commons.Triplet;

/**
 * A class to perform SVD.
 * Class attempts to reconstruct
 * a target matrix using an LU decomposition 
 * of a specified number of concepts.
 * The target matrix is assumed to have values
 * of the form (firstArgument, secondArgument).
 * @author mc2711
 *
 */
public abstract class SVD {
		
	private final int concepts;
	private final int maxIterations;
	private final boolean verbose;
			
	private final List<Triplet> inputData;
		
	private Map<Integer, Double> etas;
	
	private final SVDTrainedModel trainedModel;
	
	private final static int K_FOLD = 5;
	
	public SVD(int concepts, int maxIterations, List<Triplet> inputData,
			double lambda, boolean verbose,
			Map<String, Integer> firstArgumentTranslation, Map<String, Integer> secondArgumentTranslation,
			Set<Integer> labels){
		
		this.concepts = concepts;
		this.inputData = inputData;
		this.maxIterations = maxIterations;
		this.verbose = verbose;
				
		/* 	this is a pretty weird way to calculate the
		* 	cardinality of first arguments. the reason why we do it this way
		*	as opposed to just doing firstArgumentTranslation.size() is because
		* 	for cross validation, we only pass in a subset of the true number
		* 	of first and second argument mappings. So, using size is going to result
		* 	in issues of indexing. This way runs in O(n), and produces sometimes
		* 	too large Q & P matrices, but no complicated mappings or lookups are required
		* 	in the rest of the algorithm.
		*/
		int firstArgumentNumber = Collections.max(firstArgumentTranslation.values()) + 1;

		RealMatrix P = new BlockRealMatrix(firstArgumentNumber, concepts);
		RealMatrix firstArgumentBiases = new BlockRealMatrix(firstArgumentNumber, 1);
		
		int secondArgumentNumber = Collections.max(secondArgumentTranslation.values()) + 1;
		
		RealMatrix Q = new BlockRealMatrix(secondArgumentNumber, concepts);
		RealMatrix secondArgumentBiases = new BlockRealMatrix(secondArgumentNumber, 1);
		
		double meanRating = calculateMeanRating(this.inputData);
		
		this.trainedModel = new SVDTrainedModel(P, Q, firstArgumentBiases, secondArgumentBiases,
				lambda, meanRating, firstArgumentTranslation, secondArgumentTranslation, labels);
		
		
		initializeMatrices();
		
	}
	
	public SVD(int concepts, int maxIterations, double lambda, boolean verbose, File trainingFile) {
		
		this.concepts = concepts;
		this.maxIterations = maxIterations;
		this.verbose = verbose;
		
		Map<String, Integer> secondArgumentTranslation = new HashMap<>();
		Map<String, Integer> firstArgumentTranslation = new HashMap<>();
		Set<Integer> labels = new HashSet<>();

		this.inputData = FileParsing.parseInputFile(trainingFile,
				firstArgumentTranslation, secondArgumentTranslation, labels, true, true);

		RealMatrix P = new BlockRealMatrix(firstArgumentTranslation.size(), concepts);
		RealMatrix firstArgumentBiases = new BlockRealMatrix(firstArgumentTranslation.size(), 1);
		
		RealMatrix Q = new BlockRealMatrix(secondArgumentTranslation.size(), concepts);
		RealMatrix secondArgumentBiases = new BlockRealMatrix(secondArgumentTranslation.size(), 1);
		
		double meanRating = calculateMeanRating(this.inputData);
		
		this.trainedModel = new SVDTrainedModel(P, Q, firstArgumentBiases, secondArgumentBiases,
				lambda, meanRating, firstArgumentTranslation, secondArgumentTranslation, labels);
		
		initializeMatrices();
	}
	
	/**
	 * Perform training via SGD. Report progress via NRMSE metric.
	 */
	public void train(){
		
		if (verbose){
			System.out.println("Start Training...");
		}
		
		// check if the etas Map has been initialized correctly
		// by the subclass ...
		assert this.etas != null : "Eta Map Not Intialized!";
		
		double previousNRMSE = Double.MAX_VALUE;
		double nrmse = previousNRMSE / 2;
		
		long startTime = 0;
		
		for (int i = 0; i < maxIterations; i++){
			
			if (nrmse >= previousNRMSE){
				break;
			}
			
			 if (verbose){
				 startTime = System.currentTimeMillis();
			 }
			
			for (Triplet input : this.inputData){
				
				int first = input.first;
				int second = input.second;
				int rating = input.third;
								
				double error = rating;
				error -= trainedModel.getMeanRating();
				error -= trainedModel.getFirstArgumentBiases().getEntry(first, 0);
				error -= trainedModel.getSecondArgumentBiases().getEntry(second, 0);
				error -= trainedModel.getQ().getRowVector(second).dotProduct(trainedModel.getP().getRowVector(first));
								
				assert etas.get(rating) != null : "No eta supplied for label : " + rating;
				double etaUsed = etas.get(rating);
				
				RealVector qRowUpdate = trainedModel.getP().getRowVector(first).mapMultiply(error);
				qRowUpdate = qRowUpdate.subtract(trainedModel.getQ().getRowVector(second).mapMultiply(trainedModel.getLambda()));
				qRowUpdate.mapMultiplyToSelf(etaUsed);
				
				RealVector pRowUpdate = trainedModel.getQ().getRowVector(second).mapMultiply(error);
				pRowUpdate = pRowUpdate.subtract(trainedModel.getP().getRowVector(first).mapMultiply(trainedModel.getLambda()));
				pRowUpdate.mapMultiplyToSelf(etaUsed);
				
				double firstArgumentBiasUpdate = etaUsed * (error - trainedModel.getFirstArgumentBiases().getEntry(first, 0) * trainedModel.getLambda());				
				double secondArgumentBiasUpdate = etaUsed * (error - trainedModel.getSecondArgumentBiases().getEntry(second, 0) * trainedModel.getLambda());
				
				trainedModel.getQ().setRowVector(second, trainedModel.getQ().getRowVector(second).add(qRowUpdate));
				trainedModel.getP().setRowVector(first, trainedModel.getP().getRowVector(first).add(pRowUpdate));
				trainedModel.getFirstArgumentBiases().setEntry(first, 0, trainedModel.getFirstArgumentBiases().getEntry(first, 0) + firstArgumentBiasUpdate);
				trainedModel.getSecondArgumentBiases().setEntry(second, 0, trainedModel.getSecondArgumentBiases().getEntry(second, 0) + secondArgumentBiasUpdate);	
			}
						
			// report NRMSE to check how learning is coming along.
			previousNRMSE = nrmse;
			nrmse = 0;
			for (Triplet input : this.inputData){
				int first = input.first;
				int second = input.second;
				int rating = input.third;
				
				double update = rating;
				update -= trainedModel.getMeanRating();
				update -= trainedModel.getFirstArgumentBiases().getEntry(first, 0);
				update -= trainedModel.getSecondArgumentBiases().getEntry(second, 0);
				update -= trainedModel.getQ().getRowVector(second).dotProduct(trainedModel.getP().getRowVector(first));
				
				nrmse += Math.pow(update, 2);
				
			}
			
			// apply regularization.
			for (int first : this.trainedModel.getFirstArgumentTranslation().values()){
				nrmse += trainedModel.getLambda() * trainedModel.getP().getRowVector(first).dotProduct(trainedModel.getP().getRowVector(first));
				nrmse += trainedModel.getLambda() * Math.pow(trainedModel.getFirstArgumentBiases().getEntry(first, 0), 2);
			}
			
			for (int second : this.trainedModel.getSecondArgumentTranslation().values()){				
				nrmse += trainedModel.getLambda() * trainedModel.getQ().getRowVector(second).dotProduct(trainedModel.getQ().getRowVector(second));
				nrmse += trainedModel.getLambda() * Math.pow(trainedModel.getSecondArgumentBiases().getEntry(second, 0), 2);
			}
			
			if (verbose){
				long endTime = System.currentTimeMillis();
				double durationInSeconds = ((double) endTime - startTime) / 1000;
				System.out.printf("Iteration: %3d\tNRMSE: %.4f\tTime: %.4f secs.\n", (i + 1), nrmse, durationInSeconds);
			}
			

		}
		
		if (verbose){
			System.out.println("Training Done.");
		}
	}
	
	public Pair<Double, Integer> predict(String firstArgument, String secondArgument){
		return trainedModel.predict(firstArgument, secondArgument);
	}
	
	/**
	 * Predict a label for the given arguments passed in.
	 * @param input
	 * @return
	 */
	public Pair<Double, Integer> predict(int firstArgument, int secondArgument){
		return trainedModel.predict(firstArgument, secondArgument);
	}
	
	
	public void test(File testingFile) {
		List<Triplet> testingData = FileParsing.parseInputFile(testingFile, 
				trainedModel.getFirstArgumentTranslation(), trainedModel.getSecondArgumentTranslation(), trainedModel.getLabels(), false, false);
		test(testingData);
	}
	
	public void test(List<Triplet> testingData){
		
		List<Pair<Double, Integer>> predictions = new LinkedList<>();
		List<Integer> groundTruth = new LinkedList<>();
		
		for (Triplet triplet : testingData){
			predictions.add(predict(triplet.first, triplet.second));
		}
		
		Metrics.collapse(testingData, this.trainedModel.getLabels(), this.verbose);
		
		for (Triplet triplet : testingData){
			groundTruth.add(triplet.third);
		}
		
		Metrics.evaluatePrecisionAndRecall(predictions, groundTruth);
		
		Map<Integer, Double> labelWeights = Metrics.getWeightsOfLabels(this.etas);
		if (verbose){
			System.out.println("Label Weights: " + labelWeights);
		}
		Metrics.evaluatePointsScored(testingData, predictions, labelWeights);
	}
	
	/**
	 * Calculate mean rating.
	 * I assume the ratings are 1..n
	 * @param input
	 * @return
	 */
	public double calculateMeanRating(List<Triplet> inputData){
		double totalRatings = 0;
		double sumOfRatings = 0;
		
		for (Triplet input : inputData){
			sumOfRatings += input.third;
			totalRatings += 1;
		}
				
		return sumOfRatings / totalRatings;
	}
	
	/**
	 * Use mean rating & random values to calculate starting elements.
	 */
	private void initializeMatrices(){
		double coefficient = Math.sqrt((trainedModel.getLabels().size() - trainedModel.getMeanRating()) / concepts);
						
		for (int i = 0; i < trainedModel.getQ().getRowDimension(); i++){
			for (int j = 0; j < trainedModel.getQ().getColumnDimension(); j++){
				trainedModel.getQ().setEntry(i, j, coefficient * Math.random());
			}
		}
				
		for (int i = 0; i < trainedModel.getP().getRowDimension(); i++){
			for (int j = 0; j < trainedModel.getP().getColumnDimension(); j++){
				trainedModel.getP().setEntry(i, j, coefficient * Math.random());
			}
		}
		
		for (int i = 0; i < trainedModel.getFirstArgumentBiases().getRowDimension(); i++){
			trainedModel.getFirstArgumentBiases().setEntry(i, 0, -trainedModel.getMeanRating() + trainedModel.getLabels().size() * Math.random());
		}
				
		for (int i = 0; i < trainedModel.getSecondArgumentBiases().getRowDimension(); i++){
			trainedModel.getSecondArgumentBiases().setEntry(i, 0, -trainedModel.getMeanRating() + trainedModel.getLabels().size() * Math.random());
		}
		
	}
	
	/**
	 * Serialize the trained model for later use.
	 * @param trainedModel
	 * @param filePath
	 */
	public static void serializeModel(SVD svd, File filePath) {
		
		try (FileOutputStream fileOut = new FileOutputStream(filePath);
		ObjectOutputStream out = new ObjectOutputStream(fileOut)){
			out.writeObject(svd.trainedModel);
		}
		catch (IOException e){
			System.err.println(e);
			e.printStackTrace();
			System.exit(1);
		}
	}
	
	/**
	 * Unserialize a trained model.
	 * @param filePath
	 * @return
	 */
	public static SVDTrainedModel unserializeModel (File filePath){
		
		try (FileInputStream fileIn = new FileInputStream(filePath);
				ObjectInputStream in = new ObjectInputStream(fileIn)){
					return SVDTrainedModel.class.cast(in.readObject());
		}
		catch (Exception e){
			System.err.println(e);
			e.printStackTrace();
			System.exit(1);
		}
		
		return null;

	}
	
	public int getConcepts() {
		return concepts;
	}

	public int getMaxIterations() {
		return maxIterations;
	}

	public boolean isVerbose() {
		return verbose;
	}

	public static int getK_FOLD() {
		return K_FOLD;
	}

	public List<Triplet> getInputData() {
		return inputData;
	}

	public Map<Integer, Double> getEtas() {
		return etas;
	}

	public SVDTrainedModel getTrainedModel() {
		return trainedModel;
	}

	public void setEtas(Map<Integer, Double> etas) {
		this.etas = etas;
	}

}
