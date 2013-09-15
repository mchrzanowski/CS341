package RBM;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class RawRBM {
  // constants
  private int NUM_ITEMS = 0;
  private int NUM_USERS = 0;
  private int NUM_LABELS = 0;
  private int NUM_FEATURES = 0;

  // shared parameters
  int cdsteps;
  double epsilonvh;
  double epsilonv;
  double epsilonh;
  double weightcost;
  double momentum;
  double finalmomentum;

  private Map<Long, Integer> itemTranslation;
  private Map<Long, Integer> userTranslation;
  private Set<Integer> labels;

  private int[][] userToItemsTraining;
  private double[] itemcount;
  private double[][] itemlcount;
  private double numitems;

  private double hidprobs[];
  private double hidact[];
  private double visprobs[][];
  private double visact[][];
  private int vissoftmax[];

  private double[][][] w_vh;
  private double[][] w_v;
  private double[] w_h;

  private double[][][] wu_vh;
  private double[][] wu_v;
  private double[] wu_h;

  private double[][][] cd_vh;
  private double[][] cd_v;
  private double[] cd_h;

  private double nrmse;

  public RawRBM(int pNoOfFeatures) {
    NUM_FEATURES = pNoOfFeatures;
  }

  private void initializeTrain() {
    w_vh = new double[NUM_ITEMS][NUM_LABELS][NUM_FEATURES];
    w_v = new double[NUM_ITEMS][NUM_LABELS];
    w_h = new double[NUM_FEATURES];

    wu_vh = new double[NUM_ITEMS][NUM_LABELS][NUM_FEATURES];
    wu_v = new double[NUM_ITEMS][NUM_LABELS];
    wu_h = new double[NUM_FEATURES];

    cd_vh = new double[NUM_ITEMS][NUM_LABELS][NUM_FEATURES];
    cd_v = new double[NUM_ITEMS][NUM_LABELS];
    cd_h = new double[NUM_FEATURES];

    hidprobs = new double[NUM_FEATURES];
    hidact = new double[NUM_FEATURES];
    visprobs = new double[NUM_ITEMS][NUM_LABELS];
    visact = new double[NUM_ITEMS][NUM_LABELS];
    vissoftmax = new int[NUM_ITEMS];

    cdsteps = 1;
    nrmse = 2.0;

    // initialize weights w_vh
    for (int item = 0; item < NUM_ITEMS; ++item) {
      for (int label = 0; label < NUM_LABELS; ++label) {
        randn(w_vh[item][label]);
      }
    }

    // initialize hidden biases
    fill(w_h, 0);

    // initialize visible biases
    for (int item = 0; item < NUM_ITEMS; ++item) {
      for (int label = 0; label < NUM_LABELS; ++label) {
        double prob = itemlcount[item][label] / itemcount[item];
        if (prob != 0) {
          w_v[item][label] = Math.log(prob);
        }
      }
    }
  }

  /**
   * 
   * @param filename
   */
  private void test(String filename) {

  }

  /**
   * Train on the entire training data
   */
  private void train(String filename) {
    parseTrainingFile(filename);
    initializeTrain();

    System.out.println("training start...");
    for (int loopcount = 0; loopcount < 80; ++loopcount) {
      if (loopcount >= 10)
        cdsteps = 3 + (loopcount - 10) / 5;
      if (loopcount > 5)
        momentum = finalmomentum;
      nrmse = 0;

      long startTime = System.currentTimeMillis();

      for (int userId = 0; userId < NUM_USERS; ++userId) {
        trainSingleUser(userId);
      }
      updateWeights();

      long endTime = System.currentTimeMillis();
      System.out.printf("training loop %d in %d seconds\n", loopcount,
          (endTime - startTime) / 1000);

      if (loopcount > 8) {
        epsilonvh *= 0.92;
        epsilonv *= 0.92;
        epsilonh *= 0.92;
      } else if (loopcount > 6) {
        epsilonvh *= 0.9;
        epsilonv *= 0.9;
        epsilonh *= 0.9;
      } else if (loopcount > 2) {
        epsilonvh *= 0.78;
        epsilonv *= 0.78;
        epsilonh *= 0.78;
      }
    }
  }

  private void updateWeights() {
    // w_vh = wu_vh * momentum + epsilonvh * (cd_vh - weightcost * w_vh)
    for (int item = 0; item < NUM_ITEMS; ++item) {
      for (int label = 0; label < NUM_LABELS; ++label) {
        for (int feature = 0; feature < NUM_FEATURES; ++feature) {
          double cd = cd_vh[item][label][feature] / itemcount[item];
          if (cd != 0) {
            wu_vh[item][label][feature] = momentum
                * wu_vh[item][label][feature] + epsilonvh
                * (cd - weightcost * w_vh[item][label][feature]);
            w_vh[item][label][feature] += wu_vh[item][label][feature];
          }
        }
      }
    }

    // w_v = wu_v * momentum + epsilonv * cd_v
    for (int item = 0; item < NUM_ITEMS; ++item) {
      for (int label = 0; label < NUM_LABELS; ++label) {
        double cd = cd_v[item][label] / itemcount[item];
        if (cd != 0) {
          wu_v[item][label] = momentum * wu_v[item][label] + epsilonv * cd;
          w_v[item][label] += wu_v[item][label];
        }
      }
    }

    // w_h = wu_h * momentum + epsilonh * cd_h
    for (int feature = 0; feature < NUM_FEATURES; ++feature) {
      double cd = cd_h[feature] / numitems;
      if (cd != 0) {
        wu_h[feature] = momentum * wu_h[feature] + epsilonh * cd;
        w_h[feature] += wu_h[feature];
      }
    }
  }

  /****************************************************************************
   * RBM STUFF
   ***************************************************************************/

  private void trainSingleUser(int user) {
    int[] data = userToItemsTraining[user];
    computeHiddenGivenVisible(data, hidprobs);
    sampleHidden(hidprobs, hidact);
    accumulatePositiveCD(data, hidact);

    for (int step = 0; step < cdsteps; ++step) {
      computeVisibleGivenHidden(data, hidprobs, visprobs);
      sampleVisible(data, visprobs, visact, vissoftmax);
      computeHiddenGivenVisible(data, hidprobs);
      sampleHidden(hidprobs, hidact);
    }
    accumulateNegativeCD(data, hidact, visact, vissoftmax);
  }

  private void accumulatePositiveCD(int[] data, double[] poshidact) {
    // compute cd_vh
    for (int idx = 0; idx < data.length; ++idx) {
      int itemLabel = data[idx];
      int item = getItemFromPair(itemLabel);
      int label = getLabelFromPair(itemLabel);
      for (int feature = 0; feature < NUM_FEATURES; ++feature) {
        if (poshidact[feature] >= 1) {
          cd_vh[item][label][feature] += 1;
        }
      }
      // compute cd_v
      cd_v[item][label] += 1;
    }
    // compute cd_h
    for (int feature = 0; feature < NUM_FEATURES; ++feature) {
      cd_h[feature] += poshidact[feature];
    }
  }

  private void accumulateNegativeCD(int[] data, double[] neghidact,
      double[][] negvisact, int[] negvissoftmax) {
    for (int idx = 0; idx < data.length; ++idx) {
      int itemLabel = data[idx];
      int item = getItemFromPair(itemLabel);
      int label = getLabelFromPair(itemLabel);
      for (int feature = 0; feature < NUM_FEATURES; ++feature) {
        label = negvissoftmax[item];
        cd_vh[item][label][feature] -= 1;
      }
      // compute cd_v
      cd_v[item][label] -= negvisact[item][label];
    }
    // compute cd_h
    for (int feature = 0; feature < NUM_FEATURES; ++feature) {
      cd_h[feature] -= neghidact[feature];
    }
  }

  /**
   * Compute hidden layer given visible
   * 
   * @param first
   * @param result[NUM_FEATURES]
   */
  private void computeHiddenGivenVisible(int[] data, double[] result) {
    fill(result, 0);
    // compute w_vh * v
    for (int idx = 0; idx < data.length; ++idx) {
      int itemLabel = data[idx];
      int item = getItemFromPair(itemLabel);
      int label = getLabelFromPair(itemLabel);
      for (int feature = 0; feature < NUM_FEATURES; ++feature) {
        result[feature] += w_vh[item][label][feature];
      }
    }
    // compute h = 1.0 / (1.0 + exp(-w_vh * v - w_h))
    for (int feature = 0; feature < NUM_FEATURES; ++feature) {
      result[feature] = 1.0 / (1.0 + exp(-result[feature] - w_h[feature]));
    }
  }

  /**
   * 
   * @param hidden[NUM_HIDDEN]
   * @param activation[NUM_HIDDEN]
   */
  private void sampleHidden(double[] hidden, double[] activation) {
    fill(activation, 0);
    for (int feature = 0; feature < NUM_FEATURES; ++feature) {
      if (hidden[feature] > Math.random()) {
        activation[feature] += 1;
      }
    }
  }

  /**
   * 
   * @param hidden[NUM_FEATURES]
   * @param result[NUM_ITEMS][NUM_LABELS]
   */
  private void computeVisibleGivenHidden(int[] data, double[] hidden,
      double[][] result) {
    fill(result, 0);
    // approximate w_vh * h
    for (int idx = 0; idx < data.length; ++idx) {
      int itemLabel = data[idx];
      int item = getItemFromPair(itemLabel);
      for (int feature = 0; feature < NUM_FEATURES; ++feature) {
        // accumulate weight for sampled hidden features
        if (hidden[feature] < 1)
          continue;
        for (int label = 0; label < NUM_LABELS; ++label) {
          result[item][label] += w_vh[item][label][feature];
        }
      }
      // compute v = 1.0 / (1.0 + exp(-w_vh * h - w_v))
      double normalizer = 0;
      for (int label = 0; label < NUM_LABELS; ++label) {
        double v = 1. / (1 + exp(-result[item][label] - w_v[item][label]));
        result[item][label] = v;
        normalizer += v;
      }
      // normalize
      for (int label = 0; normalizer > 0 && label < NUM_LABELS; ++label) {
        result[item][label] /= normalizer;
      }
    }
  }

  /**
   * 
   * @param data
   * @param visible[NUM_ITEMS][NUM_LABELS]
   * @param activation[NUM_ITEMS][NUM_LABELS]
   * @param vlabels[NUM_ITEMS] - visible label e.g. 1,2,3
  */
  private void sampleVisible(int[] data, double[][] visible,
      double[][] activation, int[] vlabels) {
    fill(activation, 0);
    fill(vlabels, 0);
    // approximate w_vh * h
    for (int idx = 0; idx < data.length; ++idx) {
      int itemLabel = data[idx];
      int item = getItemFromPair(itemLabel);
      int vlabel = NUM_LABELS - 1;
      double randval = Math.random();
      for (int label = 0; label < NUM_LABELS; ++label) {
        if ((randval -= visible[item][label]) <= 0) {
          vlabel = label;
          break;
        }
      }
      vlabels[item] = vlabel;
      activation[item][vlabel] += 1;
    }
  }

  /****************************************************************************
   * HELPER METHODS
   ***************************************************************************/

  private int makeItemLabelPair(int itemId, int rating) {
    return (itemId << 4) + rating;
  }

  private int getItemFromPair(int itemLabel) {
    return (itemLabel >> 4);
  }

  private int getLabelFromPair(int itemLabel) {
    return itemLabel & 0xF;
  }

  /**
   * Read in input file. Translate raw numbers to model-specific numbers.
   * Assumes input is lines of form USER\tITEM\tRATING
   * 
   * @param trainingFile
   */
  private void parseTrainingFile(String filename) {
    this.itemTranslation = new HashMap<Long, Integer>();
    this.userTranslation = new HashMap<Long, Integer>();
    this.labels = new HashSet<Integer>();

    try {
      System.out.println("start parsing training file...");

      String line;
      HashMap<Integer, List<Integer>> userToItemsTemp = new HashMap<Integer, List<Integer>>();
      BufferedReader br = new BufferedReader(new FileReader(filename));
      while ((line = br.readLine()) != null) {
        String[] chunks = line.split("\t");
        long user = Long.parseLong(chunks[0]);
        long item = Long.parseLong(chunks[1]);
        int label = Integer.parseInt(chunks[2]) - 1;
        if (!userTranslation.containsKey(user)) {
          userTranslation.put(user, userTranslation.size());
        }
        if (!itemTranslation.containsKey(item)) {
          itemTranslation.put(item, itemTranslation.size());
        }
        if (!labels.contains(label)) {
          labels.add(label);
        }

        if (userTranslation.containsKey(user)
            && itemTranslation.containsKey(item)) {
          int userId = userTranslation.get(user);
          int itemId = itemTranslation.get(item);
          if (!userToItemsTemp.containsKey(userId)) {
            userToItemsTemp.put(userId, new ArrayList<Integer>());
          }
          List<Integer> userItems = userToItemsTemp.get(userId);
          userItems.add(makeItemLabelPair(itemId, label));
        }
      }
      br.close();

      NUM_USERS = userTranslation.size();
      NUM_ITEMS = itemTranslation.size();
      NUM_LABELS = labels.size();
      this.itemcount = new double[NUM_ITEMS];
      this.itemlcount = new double[NUM_ITEMS][NUM_LABELS];
      this.numitems = 0;

      userToItemsTraining = new int[userToItemsTemp.size()][];
      for (int userId : userToItemsTemp.keySet()) {
        List<Integer> userItems = userToItemsTemp.get(userId);
        userToItemsTraining[userId] = new int[userItems.size()];
        for (int idx = 0; idx < userItems.size(); ++idx) {
          int itemLabel = userItems.get(idx);
          int item = getItemFromPair(itemLabel);
          int label = getLabelFromPair(itemLabel);
          itemlcount[item][label]++;
          itemcount[item]++;
          numitems++;
          userToItemsTraining[userId][idx] = itemLabel;
        }
      }
      System.out.println("finished parsing training file...");

    } catch (Exception e) {
      e.printStackTrace();
      System.exit(0);
    }
  }

  /****************************************************************************
   * MATRIX HELPER
   ***************************************************************************/

  private void randn(double[] vector) {
    for (int i = 0; i < vector.length; ++i) {
      vector[i] = 0.02 * Math.random() - 0.01;
    }
  }

  private void fill(int[] vector, int value) {
    java.util.Arrays.fill(vector, value);
  }

  private void fill(double[] vector, double value) {
    java.util.Arrays.fill(vector, value);
  }

  private void fill(double[][] vector, double value) {
    for (int i = 0; i < vector.length; ++i) {
      java.util.Arrays.fill(vector[i], value);
    }
  }

  private double exp(double val) {
    final long tmp = (long) (1512775 * val + 1072632447);
    return Double.longBitsToDouble(tmp << 32);
  }

  /****************************************************************************
   * MAIN
   ***************************************************************************/

  public static void main(String[] args) {
    int numFeatures = 100;
    String trainingFileLocation = "./data/training";
    String testFileLocation = "./data/testing";

    if (args.length > 1) {
      for (int i = 0; i < args.length; i++) {
        String[] option = args[i].split("=");
        if (option[0].equals("-h")) {
          numFeatures = Integer.parseInt(option[1]);
        } else if (option[0].equals("-train")) {
          trainingFileLocation = option[1];
        } else if (option[0].equals("-test")) {
          testFileLocation = option[1];
        }
      }
    }

    long startTime = System.currentTimeMillis();
    RawRBM rbm = new RawRBM(numFeatures);
    rbm.train(trainingFileLocation);
    rbm.test(testFileLocation);
    long endTime = System.currentTimeMillis();
    System.out.printf("Runtime: %d seconds", (endTime - startTime) / 1000);

  }
}
