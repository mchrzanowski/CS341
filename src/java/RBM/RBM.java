package RBM;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;
import java.util.NavigableSet;
import java.util.Random;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import commons.FileParsing;
import commons.Metrics;
import commons.Prediction;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.DoubleMatrix3D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix3D;
import cern.jet.math.Functions;

import cern.colt.matrix.linalg.Algebra;

public class RBM {

	private static final Algebra LinearAlgebra = Algebra.DEFAULT;

	private final boolean verbose;

	private final int batchSize;
	private final int noOfFeatures;
	private final int ITEM_COUNT;
	private final int SOFTMAX;

	private final double epsilonw;
	private final double epsilonvb;
	private final double epsilonhb;
	private final double weight_cost;
	private final double momentum;
	private final double final_momentum;

	private final DoubleMatrix2D item_rating_count;
	private final DoubleMatrix2D visible_biases;
	private final DoubleMatrix1D hidden_biases;
	private final DoubleMatrix3D weights;

	private final int iterationsToPerform;

	private final Map<String, Integer> itemTranslation;
	private final Map<String, Integer> userTranslation;
	private NavigableMap<Integer, List<Pair>> allUserToItemsTraining;
	private NavigableMap<Integer, List<Pair>> userToItemsTraining;

	private final Set<Integer> labels;

	private Random rng;
	private File testingFile;
	private File trainingFile;

	public RBM(int batchSize, int features, boolean verbose, 
			File trainingFile, File testingFile, int iterationsToPerform) {
		this.testingFile = testingFile;
		this.trainingFile = trainingFile;
		this.batchSize = batchSize;
		this.noOfFeatures = features;
		this.verbose = verbose;
		this.iterationsToPerform = iterationsToPerform;

		this.itemTranslation = new HashMap<>();
		this.userTranslation = new HashMap<>();
		this.labels = new HashSet<>();

		this.allUserToItemsTraining =
			FileParsing.parseUserItemFile(
				trainingFile, this.itemTranslation,
				this.userTranslation, this.labels, true, true);
		this.userToItemsTraining = RBM.sampledData(
			this.allUserToItemsTraining);

		ITEM_COUNT = itemTranslation.size();
		SOFTMAX = labels.size();

		weights = new DenseDoubleMatrix3D(ITEM_COUNT, SOFTMAX,
				this.noOfFeatures);
		item_rating_count = new DenseDoubleMatrix2D(ITEM_COUNT, SOFTMAX);

		visible_biases = new DenseDoubleMatrix2D(ITEM_COUNT, SOFTMAX);
		hidden_biases = new DenseDoubleMatrix1D(noOfFeatures);

		this.rng = new Random();

		epsilonw = 0.001;
		epsilonvb = 0.008;
		epsilonhb = 0.006;
		weight_cost = 0.0001;
		momentum = 0.8;
		final_momentum = 0.9;

		initialize();
	}

	private void initialize() {

		// weight initialization.
		for (int i = 0; i < weights.slices(); i++) {
			for (int j = 0; j < weights.rows(); j++) {
				for (int k = 0; k < weights.columns(); k++) {
					weights.setQuick(i, j, k, 0.02 * rng.nextGaussian() - 0.01);
				}
			}
		}

		initializeVisibleBiases();

		if (verbose) {
			System.out.println("Batch Size:\t" + this.batchSize);
			System.out.println("Features:\t" + this.noOfFeatures);
			System.out.println("Labels:\t\t" + this.SOFTMAX);
			System.out.println("Items:\t\t" + this.ITEM_COUNT);
			System.out.println("Iterations:\t" + this.iterationsToPerform);
		}

	}
	
	
	private void initializeVisibleBiases() {
		for (int user : this.userToItemsTraining.keySet()) {
			for (Pair pair : userToItemsTraining.get(user)) {
				item_rating_count.setQuick(pair.item, pair.rating,
						item_rating_count.getQuick(pair.item, pair.rating) + 1);
			}
		}

		DoubleMatrix2D non_zero_item_rating_count = item_rating_count.copy();
		non_zero_item_rating_count.assign(Functions.equals(0));
		non_zero_item_rating_count.assign(item_rating_count, Functions.plus);

		DoubleMatrix1D normalization = getRowNorm(item_rating_count);

		for (int i = 0; i < normalization.size(); i++) {
			if (normalization.getQuick(i) == 0) {
				normalization.setQuick(i, 1);
			}
		}

		for (int i = 0; i < non_zero_item_rating_count.rows(); i++) {
			for (int j = 0; j < non_zero_item_rating_count.columns(); j++) {
				visible_biases.setQuick(i, j,
						non_zero_item_rating_count.getQuick(i, j)
								/ normalization.getQuick(i));
			}
		}

		visible_biases.assign(Functions.log);
		DoubleMatrix2D activations = new DenseDoubleMatrix2D(ITEM_COUNT,
				SOFTMAX);

		for (int i = 0; i < item_rating_count.rows(); i++) {
			for (int j = 0; j < item_rating_count.columns(); j++) {
				if (item_rating_count.getQuick(i, j) > 0) {
					activations.setQuick(i, j, 1);
				}
			}
		}
		visible_biases.assign(activations, Functions.mult);

	}

	public TrainUserResult trainUser(Integer user, int cd_steps,
			DoubleMatrix2D neg_hidden_probs,
			DoubleMatrix1D negative_visual_softmax, DoubleMatrix2D nvp2,
			DoubleMatrix1D hidden_units) {

		nvp2.assign(0);
		neg_hidden_probs.assign(0);
		negative_visual_softmax.assign(0);
		hidden_units.assign(0);

		double nrmse = 0.0;
		double ntrain = 0.0;

		DoubleMatrix1D negative_hidden_states = new DenseDoubleMatrix1D(
				noOfFeatures);

		for (Pair pair : userToItemsTraining.get(user)) {
			int item = pair.item;
			int rating = pair.rating;
			hidden_units.assign(weights.viewSlice(item).viewRow(rating),
					Functions.plus);
		}

		hidden_units.assign(hidden_biases, Functions.plus);
		sigmoid(hidden_units);
		DoubleMatrix1D positive_hidden_states = new DenseDoubleMatrix1D(
				this.noOfFeatures);

		for (int i = 0; i < positive_hidden_states.size(); i++) {
			if (hidden_units.getQuick(i) > rng.nextDouble()) {
				positive_hidden_states.setQuick(i, 1);
			}
		}

		DoubleMatrix1D current_positive_hidden_states = positive_hidden_states;

		for (Pair pair : userToItemsTraining.get(user)) {
			int item = pair.item;
			int rating = pair.rating;
			nvp2.viewRow(item).assign(
					LinearAlgebra.mult(weights.viewSlice(item), hidden_units),
					Functions.plus); // Prob issue
			nvp2.viewRow(item).assign(visible_biases.viewRow(item),
					Functions.plus);
			sigmoid(nvp2.viewRow(item));

			for (int i = 0; i < nvp2.viewRow(item).size(); i++) {
				nvp2.viewRow(item).setQuick(i, nvp2.viewRow(item).getQuick(i));
			}
			double normalization = nvp2.viewRow(item).aggregate(Functions.plus,
					Functions.identity);
			nvp2.viewRow(item).assign(Functions.div(normalization));

			double expectedV = 0;
			for (int r = 0; r < SOFTMAX; r++) {
				expectedV += r * nvp2.getQuick(item, r);
			}

			double vdelta = rating - expectedV;
			nrmse += Math.pow(vdelta, 2);
		}

		ntrain += userToItemsTraining.get(user).size();
		int step = 0;
		while (step < cd_steps) {
			step += 1;
			boolean final_step = (step >= cd_steps);

			// POSITIVE CD calculation reconstruct the visible units
			for (Pair pair : userToItemsTraining.get(user)) {
				int item = pair.item;

				neg_hidden_probs
						.viewRow(item)
						.assign(LinearAlgebra.mult(weights.viewSlice(item),
								current_positive_hidden_states), Functions.plus);
				neg_hidden_probs.viewRow(item).assign(
						visible_biases.viewRow(item), Functions.plus);
				sigmoid(neg_hidden_probs.viewRow(item));

				for (int i = 0; i < neg_hidden_probs.viewRow(item).size(); i++) {
					neg_hidden_probs.viewRow(item).setQuick(i,
							neg_hidden_probs.viewRow(item).getQuick(i));
				}
				double normalization = neg_hidden_probs.viewRow(item)
						.aggregate(Functions.plus, Functions.identity);
				neg_hidden_probs.viewRow(item).assign(
						Functions.div(normalization));

				double rand_value = rng.nextDouble();
				negative_visual_softmax.setQuick(item, SOFTMAX - 1);
				for (int r = 0; r < SOFTMAX; r++) {
					rand_value -= neg_hidden_probs.getQuick(item, r);
					if (rand_value <= 0) {
						negative_visual_softmax.setQuick(item, r);
						break;
					}
				}
			}

			// NEGATIVE CD calculation. AKA create the hidden layer
			hidden_units.assign(0);
			for (Pair pair : userToItemsTraining.get(user)) {
				int item = pair.item;
				hidden_units.assign(
						weights.viewSlice(item).viewRow(
								(int) negative_visual_softmax.getQuick(item)),
						Functions.plus); // Prob issue
			}

			hidden_units.assign(hidden_biases, Functions.plus);
			sigmoid(hidden_units);

			for (int i = 0; i < negative_hidden_states.size(); i++) {
				if (hidden_units.getQuick(i) > rng.nextDouble()) {
					negative_hidden_states.setQuick(i, 1);
				} else {
					negative_hidden_states.setQuick(i, 0);
				}
			}

			if (!final_step) {
				current_positive_hidden_states = negative_hidden_states.copy();
				neg_hidden_probs.assign(0);
			}
		}

		TrainUserResult trainUserResult = new TrainUserResult();
		trainUserResult.negative_hidden_states = negative_hidden_states;
		trainUserResult.positive_hidden_states = positive_hidden_states;
		trainUserResult.items = new ArrayList<>();

		for (Pair pair : userToItemsTraining.get(user)) {
			int item = pair.item;
			int rating = pair.rating;
			ItemData itemData = new ItemData();
			itemData.item = item;
			itemData.rating = rating;
			itemData.reconstructed_rating = (int) negative_visual_softmax
					.getQuick(item);
			trainUserResult.items.add(itemData);
		}

		trainUserResult.nrmse = nrmse;
		trainUserResult.ntrain = ntrain;

		return trainUserResult;
	}

	public UpdateWeightResult updateWeights(List<TrainUserResult> emissions,
			DoubleMatrix3D incremental_CD,
			DoubleMatrix1D hidden_biases_increment,
			DoubleMatrix2D visual_bias_increment, double EpsilonW,
			double EpsilonVB, double EpsilonHB, double Momentum,
			DoubleMatrix3D positive_CD, DoubleMatrix3D negative_CD,
			DoubleMatrix1D positive_hidden_activations,
			DoubleMatrix1D negative_hidden_activations,
			DoubleMatrix2D positive_visual_activations,
			DoubleMatrix2D negative_visual_activations,
			DoubleMatrix1D item_count,
			Map<Integer, Double> etas_vh,
			Map<Integer, Double> etas_v) {
		positive_CD.assign(0);
		negative_CD.assign(0);

		positive_hidden_activations.assign(0);
		negative_hidden_activations.assign(0);

		positive_visual_activations.assign(0);
		negative_visual_activations.assign(0);

		item_count.assign(0);

		double nrmse = 0.0;
		double ntrain = 0.0;

		for (TrainUserResult emission : emissions) {
			nrmse += emission.nrmse;
			ntrain += emission.ntrain;

			negative_hidden_activations.assign(emission.negative_hidden_states,
					Functions.plus);
			positive_hidden_activations.assign(emission.positive_hidden_states,
					Functions.plus);

			for (ItemData itemData : emission.items) {
				item_count.setQuick(itemData.item,
						item_count.getQuick(itemData.item) + 1);
				positive_visual_activations.setQuick(itemData.item,
						itemData.rating, positive_visual_activations.getQuick(
								itemData.item, itemData.rating) + 1);
				negative_visual_activations.setQuick(itemData.item,
						itemData.reconstructed_rating,
						negative_visual_activations.getQuick(itemData.item,
								itemData.reconstructed_rating) + 1);
				positive_CD
						.viewSlice(itemData.item)
						.viewRow(itemData.rating)
						.assign(emission.positive_hidden_states, Functions.plus);
				negative_CD
						.viewSlice(itemData.item)
						.viewRow(itemData.reconstructed_rating)
						.assign(emission.negative_hidden_states, Functions.plus);
			}
		}

		int numcases = emissions.size();

		for (int item = 0; item < this.ITEM_COUNT; item++) {
			double CDp, CDn;
			if (item_count.getQuick(item) == 0)
				continue;

			for (int feature = 0; feature < this.noOfFeatures; feature++) {
				for (int r = 0; r < this.SOFTMAX; r++) {
					CDp = positive_CD.getQuick(item, r, feature);
					CDn = negative_CD.getQuick(item, r, feature);
					if ((CDp != 0) || (CDn != 0)) {
						CDp /= item_count.getQuick(item);
						CDn /= item_count.getQuick(item);
					}

					incremental_CD.setQuick(item, r, feature, Momentum * incremental_CD.getQuick(item, r, feature) + 
							EpsilonW * ((CDp - CDn) - weight_cost * weights.getQuick(item, r,feature)));
					weights.setQuick(item, r,feature, weights.getQuick(item, r, feature) +
							incremental_CD.getQuick(item, r, feature));
				}
			}

			for (int r = 0; r < SOFTMAX; r++) {
				if (positive_visual_activations.getQuick(item, r) != 0 || negative_visual_activations.getQuick(item, r) != 0) {
					positive_visual_activations.setQuick(item, r,
							positive_visual_activations.getQuick(item, r) / item_count.getQuick(item));
					negative_visual_activations.setQuick(item, r,
							negative_visual_activations.getQuick(item, r) / item_count.getQuick(item));
					visual_bias_increment.setQuick(item, r, Momentum * visual_bias_increment.getQuick(item, r)
							+ EpsilonVB * ((positive_visual_activations.getQuick(item, r) - negative_visual_activations.getQuick(item, r))));
					visible_biases.setQuick(item, r, visible_biases.getQuick(item, r)
							+ visual_bias_increment.getQuick(item, r));
				}
			}
		}

		for (int feature = 0; feature < this.noOfFeatures; feature++) {
			if (positive_hidden_activations.getQuick(feature) != 0 || negative_hidden_activations.getQuick(feature) != 0) {
				positive_hidden_activations.setQuick(feature,
						positive_hidden_activations.getQuick(feature) / numcases);
				negative_hidden_activations.setQuick(feature,
						negative_hidden_activations.getQuick(feature) / numcases);
				hidden_biases_increment.setQuick(feature, Momentum * hidden_biases_increment.getQuick(feature)
						+ EpsilonHB * ((positive_hidden_activations.getQuick(feature) - negative_hidden_activations.getQuick(feature))));
				hidden_biases.setQuick(feature, hidden_biases.getQuick(feature)
						+ hidden_biases_increment.getQuick(feature));
			}
		}

		UpdateWeightResult updateWeightResult = new UpdateWeightResult();
		updateWeightResult.nrmse_update = nrmse;
		updateWeightResult.ntrain_update = ntrain;
		return updateWeightResult;
	}

	public void train() {

		final DoubleMatrix3D incremental_CD = new DenseDoubleMatrix3D(
				ITEM_COUNT, SOFTMAX, this.noOfFeatures);
		final DoubleMatrix1D hidden_biases_increment = new DenseDoubleMatrix1D(
				this.noOfFeatures);
		final DoubleMatrix2D visual_bias_increment = new DenseDoubleMatrix2D(
				this.ITEM_COUNT, this.SOFTMAX);

		final DoubleMatrix2D nvp2 = new DenseDoubleMatrix2D(this.ITEM_COUNT,
				SOFTMAX);
		final DoubleMatrix2D neg_hidden_probs = new DenseDoubleMatrix2D(
				this.ITEM_COUNT, this.SOFTMAX);
		final DoubleMatrix1D negative_visual_softmax = new DenseDoubleMatrix1D(
				this.ITEM_COUNT);

		final DoubleMatrix3D positive_CD = new DenseDoubleMatrix3D(ITEM_COUNT,
				SOFTMAX, noOfFeatures);
		final DoubleMatrix3D negative_CD = new DenseDoubleMatrix3D(ITEM_COUNT,
				SOFTMAX, noOfFeatures);

		final DoubleMatrix1D positive_hidden_activations = new DenseDoubleMatrix1D(
				this.noOfFeatures);
		final DoubleMatrix1D negative_hidden_activations = new DenseDoubleMatrix1D(
				this.noOfFeatures);

		final DoubleMatrix2D positive_visual_activations = new DenseDoubleMatrix2D(
				this.ITEM_COUNT, this.SOFTMAX);
		final DoubleMatrix2D negative_visual_activations = new DenseDoubleMatrix2D(
				this.ITEM_COUNT, this.SOFTMAX);

		final DoubleMatrix1D item_count = new DenseDoubleMatrix1D(
				this.ITEM_COUNT);
		final DoubleMatrix1D hidden_units = new DenseDoubleMatrix1D(
				this.noOfFeatures);

		double EpsilonW = epsilonw;
		double EpsilonVB = epsilonvb;
		double EpsilonHB = epsilonhb;
		double Momentum = momentum;
		
		Map<Integer, Double> etas_vh = new HashMap<>(4);
		etas_vh.put(0, 0.001);
		etas_vh.put(1, 0.001);
		etas_vh.put(2, 0.001);
		etas_vh.put(3, 0.001);

		Map<Integer, Double> etas_v = new HashMap<>(4);
		etas_v.put(0, 0.008);
		etas_v.put(1, 0.008);
		etas_v.put(2, 0.008);
		etas_v.put(3, 0.008);

		double nrmse = 2.0;

		int cd_steps = 1;
		int ntrain = 0;

		for (int iteration = 0; iteration < this.iterationsToPerform; iteration++) {
			long startTime = System.currentTimeMillis();

			//if(iteration >= 10) {
			//	cd_steps = 3 + (iteration - 10) / 5;
			//}

			ntrain = 0;
			nrmse = 0;

			//if (iteration > 5) Momentum = final_momentum;

			this.userToItemsTraining = RBM.sampledData(this.allUserToItemsTraining);
			List<Integer> user_keys = new ArrayList<>(this.userToItemsTraining.keySet());
			Collections.sort(user_keys);

			int number_of_batches = (int) Math.ceil(((double) user_keys.size())
					/ this.batchSize);
			for (int batch = 0; batch < number_of_batches; batch++) {
				int startIndex = batch * this.batchSize;
				int endIndex = Math.min(userToItemsTraining.size(), 
						(batch + 1) * this.batchSize);

				List<TrainUserResult> emissions = new LinkedList<>();
				for (int i = startIndex; i < endIndex; i++) {
					emissions.add(trainUser(user_keys.get(i), cd_steps,
							neg_hidden_probs, negative_visual_softmax, nvp2,
							hidden_units));
				}

				UpdateWeightResult updateWeightResult = updateWeights(
						emissions, incremental_CD, hidden_biases_increment,
						visual_bias_increment, EpsilonW, EpsilonVB, EpsilonHB,
						Momentum, positive_CD, negative_CD,
						positive_hidden_activations,
						negative_hidden_activations,
						positive_visual_activations,
						negative_visual_activations, item_count, etas_vh, etas_v);
				nrmse += updateWeightResult.nrmse_update;
				ntrain += updateWeightResult.ntrain_update;

			}

			nrmse = Math.pow(((float) nrmse / ntrain), 0.5);
			long endTime = System.currentTimeMillis();
			double duration = ((double) endTime - startTime) / 1000;
			System.out.format("Loop: %3d\tNRMSE: %.9f\tDuration: %.2f secs.\n",
					iteration + 1, nrmse, duration);

			if (iteration > 16) {
				EpsilonW *= 0.92;
				EpsilonVB *= 0.92;
				EpsilonHB *= 0.92;

			} else if (iteration > 8) {
				EpsilonW *= 0.9;
				EpsilonVB *= 0.9;
				EpsilonHB *= 0.9;

			} else if (iteration > 4) {
				EpsilonW *= 0.78;
				EpsilonVB *= 0.78;
				EpsilonHB *= 0.78;

			}

			//test(trainingFile);
			test(testingFile);
		}
	}

	public List<Prediction> predict(NavigableMap<Integer, List<Pair>> userToItemsTest) {

		List<Prediction> predictions = new LinkedList<>();

		DoubleMatrix2D neg_hidden_probs = new DenseDoubleMatrix2D(
				this.ITEM_COUNT, this.SOFTMAX);
		DoubleMatrix1D hidden_units = new DenseDoubleMatrix1D(this.noOfFeatures);
		DoubleMatrix1D pos_hidden_probs = new DenseDoubleMatrix1D(
				this.noOfFeatures);

		for (int user : userToItemsTest.navigableKeySet()) {

			neg_hidden_probs.assign(0);
			hidden_units.assign(0);
			pos_hidden_probs.assign(0);

			for (Pair pair : userToItemsTraining.get(user)) {
				int item = pair.item;
				int rating = pair.rating;

				DoubleMatrix1D tmpitemratingweights = weights.viewSlice(item)
						.viewRow(rating);
				for (int i = 0; i < tmpitemratingweights.size(); i++) {
					hidden_units.setQuick(i, hidden_units.getQuick(i)
							+ tmpitemratingweights.getQuick(i));
				}
			}

			hidden_units.assign(hidden_biases, Functions.plus);
			sigmoid(hidden_units);
			
			Collections.sort(userToItemsTest.get(user));
			for (Pair pair : userToItemsTest.get(user)) {
				int item = pair.item;
				neg_hidden_probs.viewRow(item).assign(
						LinearAlgebra.mult(weights.viewSlice(item),
								hidden_units), Functions.plus);

				neg_hidden_probs.viewRow(item).assign(
						visible_biases.viewRow(item), Functions.plus);
				sigmoid(neg_hidden_probs.viewRow(item));

				for (int i = 0; i < neg_hidden_probs.viewRow(item).size(); i++) {
					neg_hidden_probs.viewRow(item).setQuick(i,
							neg_hidden_probs.viewRow(item).getQuick(i));
				}

				double normalization = neg_hidden_probs.viewRow(item)
						.aggregate(Functions.plus, Functions.identity);

				for (int i = 0; i < neg_hidden_probs.viewRow(item).size(); i++) {
					neg_hidden_probs.viewRow(item).setQuick(
							i,
							neg_hidden_probs.viewRow(item).getQuick(i)
									/ normalization);
				}
			}

			for (Pair pair : userToItemsTest.get(user)) {
				int item = pair.item;

				double prediction = 0;
				for (int r = 0; r < SOFTMAX; r++) {
					prediction += r * neg_hidden_probs.getQuick(item, r);
				}

				int bucketedPrediction = (int) Math.round(prediction);

				// clip prediction into range of labels.
				if (bucketedPrediction < Collections.min(labels)) {
					bucketedPrediction = Collections.min(labels);
				} else if (bucketedPrediction > Collections.max(labels)) {
					bucketedPrediction = Collections.max(labels);
				}

				predictions.add(new Prediction(pair, bucketedPrediction));

			}
		}

		return predictions;

	}

	/**
	 * Measure predictions on new input in the file passed as an argument.
	 * 
	 * @param testFile
	 */
	public void test(File testFile) {

		NavigableMap<Integer, List<Pair>> userToItemsTest = FileParsing.parseUserItemFile(testFile, this.itemTranslation,
				this.userTranslation, this.labels, false, true);
		test(userToItemsTest);
	}

	private void test(NavigableMap<Integer, List<Pair>> data) {
		List<Prediction> predictions = predict(data);

		Map<Integer, Double> labelWeights = Metrics.getWeightsOfLabels(this.allUserToItemsTraining);
		if (verbose) {
			System.out.println("Label Weights: " + labelWeights);
		}

		Metrics.collapse(predictions, this.labels, this.verbose);
		Metrics.evaluatePrecisionAndRecall(predictions);
		Metrics.evaluatePointsScored(predictions, labelWeights);
	}


	/**
	 * @param args
	 * @throws ParseException
	 */
	public static void main(String[] args) throws ParseException {

		long startTime = System.currentTimeMillis();
		Options options = new Options();
		options.addOption("b", true, "Batch size.");
		options.addOption("f", true, "Features");
		options.addOption("n", true, "Iteration number");
		options.addOption("tr", "train", true, "Training File.");
		options.addOption("te", "test", true, "Testing File.");
		options.addOption("v", false, "Verbose mode.");

		CommandLineParser parser = new GnuParser();
		CommandLine commandLine = parser.parse(options, args);

		int batchSize = 100;
		if (commandLine.hasOption("b")) {
			batchSize = Integer.parseInt(commandLine.getOptionValue("b"));
		}

		int features = 100;
		if (commandLine.hasOption("f")) {
			features = Integer.parseInt(commandLine.getOptionValue("f"));
		}

		int iterations = 80;
		if (commandLine.hasOption("n")) {
			iterations = Integer.parseInt(commandLine.getOptionValue("n"));
		}

		boolean verbose = false;
		if (commandLine.hasOption("v")) {
			verbose = true;
		}

		File trainingFile = new File(commandLine.getOptionValue("tr"));
		File testingFile = new File(commandLine.getOptionValue("te"));

		RBM rbm = new RBM(batchSize, features, verbose, trainingFile,
				testingFile, iterations);
		rbm.train();

		System.out.println("....Training Error.......");
		rbm.test(trainingFile);

		System.out.println("....Test Error.......");
		rbm.test(testingFile);

		long endTime = System.currentTimeMillis();
		System.out.println("Runtime: "
				+ (((double) endTime - startTime) / 1000) + " seconds.");

	}

	private DoubleMatrix1D getRowNorm(DoubleMatrix2D matrix2D) {
		DoubleMatrix1D matrix2DRowNorms = new DenseDoubleMatrix1D(
				matrix2D.rows());
		for (int i = 0; i < matrix2D.rows(); i++) {
			matrix2DRowNorms.setQuick(
					i,
					matrix2D.viewRow(i).aggregate(Functions.plus,
							Functions.identity));
		}
		return matrix2DRowNorms;
	}

	/**
	 * Sigmoid. f(x) = 1 / (1 + exp(-x))
	 * 
	 * @param matrix1D
	 */
	private void sigmoid(DoubleMatrix1D matrix1D) {
		matrix1D.assign(Functions.chain(
				Functions.inv,
				Functions.chain(Functions.plus(1),
						Functions.chain(Functions.inv, Functions.exp))));
	}

	public boolean isVerbose() {
		return verbose;
	}

	public Map<String, Integer> getItemTranslation() {
		return itemTranslation;
	}

	public Map<String, Integer> getUserTranslation() {
		return userTranslation;
	}

	public NavigableMap<Integer, List<Pair>> getUserToItemsTraining() {
		return userToItemsTraining;
	}

	public Set<Integer> getLabels() {
		return labels;
	}

	public static NavigableMap<Integer, List<Pair>> sampledData(NavigableMap<Integer, List<Pair>> inputData){
			NavigableMap<Integer, List<Pair>> sampledData = new TreeMap<>();
		for(int user : inputData.keySet()) {
			List<Pair> items = inputData.get(user);
			List<Pair> list1 = new ArrayList<Pair>();
			List<Pair> list2 = new ArrayList<Pair>();
			for(Pair item : items) {
				if(item.rating == 0) {
					list1.add(item);
				} else {
					list2.add(item);
				}
			}
			List<Pair> newitems = new ArrayList<Pair>();
			Collections.shuffle(list1);
			Collections.shuffle(list2);
			int size = Math.min(list1.size(), list2.size());
			newitems.addAll(list1.subList(0, size));
			newitems.addAll(list2.subList(0, size));
			sampledData.put(user, newitems);
		}
		return sampledData;
	}

}