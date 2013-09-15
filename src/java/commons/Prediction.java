package commons;

public class Prediction {

  public int bucketedPrediction;
  public double prediction;

  public Prediction(double prediction) {
    this.prediction = prediction;
    this.bucketedPrediction = (int) Math.round(prediction);
  }

  public Prediction(double prediction, int bucketedPrediction) {
    this.prediction = prediction;
    this.bucketedPrediction = bucketedPrediction;
  }

  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + bucketedPrediction;
    long temp;
    temp = Double.doubleToLongBits(prediction);
    result = prime * result + (int) (temp ^ (temp >>> 32));
    return result;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj)
      return true;
    if (obj == null)
      return false;
    if (getClass() != obj.getClass())
      return false;
    Prediction other = (Prediction) obj;
    if (bucketedPrediction != other.bucketedPrediction)
      return false;
    if (Double.doubleToLongBits(prediction) != Double
        .doubleToLongBits(other.prediction))
      return false;
    return true;
  }

}