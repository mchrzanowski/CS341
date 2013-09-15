package SVD;

import java.io.Serializable;
import java.util.Collections;
import java.util.Map;
import java.util.Set;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math.linear.RealMatrix;

import commons.Constants;

/**
 * Container class to perform predictions.
 * @author mc2711
 *
 */
public class SVDTrainedModel implements Serializable {

	private static final long serialVersionUID = 1L;
	private final RealMatrix P;
	private final RealMatrix Q;
	
	private final RealMatrix firstArgumentBiases;
	private final RealMatrix secondArgumentBiases;
	
	private final double lambda;
	private final double meanRating;

	private final Map<String, Integer> firstArgumentTranslation;
	private final Map<String, Integer> secondArgumentTranslation;
	
	private final Set<Integer> labels;
	
	public SVDTrainedModel(RealMatrix P, RealMatrix Q, RealMatrix firstArgumentBiases,
			RealMatrix secondArgumentBiases, double lambda, double meanRating,
			Map<String, Integer> firstArgumentTranslation,
			Map<String, Integer> secondArgumentTranslation,
			Set<Integer> labels){
		this.P = P;
		this.Q = Q;
		this.firstArgumentBiases = firstArgumentBiases;
		this.secondArgumentBiases = secondArgumentBiases;
		this.lambda = lambda;
		this.meanRating = meanRating;
		this.firstArgumentTranslation = firstArgumentTranslation;
		this.secondArgumentTranslation = secondArgumentTranslation;
		this.labels = labels;
	}
	
	public Pair<Double, Integer> predict(String firstArgument, String secondArgument){
		int first = Constants.UNKNOWN;
		int second = Constants.UNKNOWN;
		
		if (firstArgumentTranslation.containsKey(firstArgument)){
			first = firstArgumentTranslation.get(firstArgument);
		}
		if (secondArgumentTranslation.containsKey(secondArgument)){
			second = secondArgumentTranslation.get(secondArgument);
		}
		
		return predict(first, second);
	}
	
	/**
	 * Predict a label for the given arguments passed in.
	 * @param input
	 * @return
	 */
	public Pair<Double, Integer> predict(int first, int second){
		
		double prediction = this.meanRating;
		
		// if we've seen the first argument, apply the argument's learned bias.
		if (first != Constants.UNKNOWN){
			prediction += firstArgumentBiases.getEntry(first, 0);
		}
		
		// ditto for the second argument bias.
		if (second != Constants.UNKNOWN){
			prediction += secondArgumentBiases.getEntry(second, 0);
		}
		if (first != Constants.UNKNOWN && second != Constants.UNKNOWN){
			prediction += Q.getRowVector(second).dotProduct(P.getRowVector(first));
		}
		
		int bucketedPrediction = (int) Math.round(prediction);
				
		// clip prediction into range of prediction labels
		if (bucketedPrediction < Collections.min(labels)){
			bucketedPrediction = Collections.min(labels);
		}
		else if (bucketedPrediction > Collections.min(labels)){
			bucketedPrediction = Collections.min(labels) + 1;
		}
		
		return Pair.of(prediction, bucketedPrediction);
	}

	public Map<String, Integer> getSecondArgumentTranslation() {
		return secondArgumentTranslation;
	}

	public Map<String, Integer> getFirstArgumentTranslation() {
		return firstArgumentTranslation;
	}

	public static long getSerialversionuid() {
		return serialVersionUID;
	}

	public RealMatrix getP() {
		return P;
	}

	public RealMatrix getQ() {
		return Q;
	}

	public RealMatrix getFirstArgumentBiases() {
		return firstArgumentBiases;
	}

	public RealMatrix getSecondArgumentBiases() {
		return secondArgumentBiases;
	}

	public double getLambda() {
		return lambda;
	}

	public double getMeanRating() {
		return meanRating;
	}

	public Set<Integer> getLabels() {
		return labels;
	}

}
