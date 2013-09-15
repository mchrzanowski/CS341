package RBM;

import java.util.List;

import cern.colt.matrix.DoubleMatrix1D;

class TrainUserResult {
    DoubleMatrix1D negative_hidden_states; 
    DoubleMatrix1D positive_hidden_states;
    List<ItemData> items;
    double nrmse;
    double ntrain;
}