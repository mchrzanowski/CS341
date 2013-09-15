package Regression;

import java.math.BigDecimal;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class WalmartQuery implements Comparable<WalmartQuery> {

  public static final double[] clickPDF = new double[] { 0.241175721556,
      0.130981735144, 0.0892994129563, 0.0713992135027, 0.0570851106386,
      0.0494814922847, 0.0441505114279, 0.0403570151304, 0.0361282328143,
      0.033366522917, 0.0309776498884, 0.0295540602239, 0.0281801065337,
      0.0269866231653, 0.0262979210055, 0.0274620860907, 0.00267735045824,
      0.00261067851026, 0.00254907391989, 0.00251444697622, 0.00241471655241,
      0.00233489360492, 0.00227514704567, 0.00226715992487, 0.00216866014504,
      0.00211856569553, 0.00208005377769, 0.00205942239314, 0.00208058464373,
      0.00205650262995, 0.00206755429559, 0.00219426236611, 2.53367880486e-05,
      2.45163587213e-05, 2.35270174737e-05, 2.34063661021e-05,
      2.44439678983e-05, 2.35994082967e-05, 2.32615844561e-05,
      2.3985492686e-05, 2.4709400916e-05, 2.29720211641e-05, 2.16207258015e-05,
      2.32857147304e-05, 2.26583275978e-05, 2.35994082967e-05,
      2.16931166245e-05, 2.33339752791e-05, 2.14518138812e-05,
      2.19344193678e-05, 2.13070322352e-05, 2.24652854031e-05,
      2.10898597662e-05, 2.08485570229e-05, 2.11381203149e-05,
      2.21033312881e-05, 2.36717991197e-05, 2.41785348807e-05,
      2.63019990219e-05, 2.5071355031e-05 };

  String visitorid;
  String rawquery;
  int[] shownitems;
  int[] clickeditems;
  double[][] svdinputs;
  
  public int find(int[] array, int value) {
    for (int i = 0; i < array.length; i++) {
      if (array[i] == value) {
        return i;
      }
    }
    return -1;
  }

  public double[] getRatings() {
    double[] ratings = new double[shownitems.length];
    Arrays.fill(ratings, 1.0);
    for (int i = 0; i < shownitems.length; ++i) {
      int item = shownitems[i];
      if (find(clickeditems, item) >= 0)
        ratings[i] = 2.0;
    }
    return ratings;
  }
 
  public double getPredictedScore(int[] ordering) {
    double score = 0;
    for (int i = 0; i < ordering.length; ++i) {
      int item = shownitems[ordering[i]];
      if(find(clickeditems, item) >= 0) {
        score += clickPDF[i];
      }
    }
    return score;    
  }

  public double getBaselineScore() {
    
    double score = 0;
    for (int i = 0; i < shownitems.length; ++i) {
      int item = shownitems[i];
      if (find(clickeditems, item) >= 0) {
        score += clickPDF[i];
      }
    }
    
    return score;
  }

  public double getMaxScore() {
    int clickedAndShownItems = 0;
    for (int i = 0; i < shownitems.length; ++i) {
      int item = shownitems[i];
      if (find(clickeditems, item) >= 0){
        clickedAndShownItems++;
      }
    }
    
    double score = 0;
    for (int i = 0; i < clickedAndShownItems; i++){
      score += clickPDF[i];
    }
    
    return score;
  }

  public double getSpecificityScore(Map<Integer, String> itemLookup) {
    if(shownitems.length == 0) return 0;
    double score = 0;
    Set<String> queryTokens = new HashSet<String>(Arrays.asList(rawquery.split("\\s+")));
    for(int itemId : shownitems) {
      String itemName = itemLookup.get(itemId);
      Set<String> itemTokens = new HashSet<String>(Arrays.asList(itemName.split("\\s+")));
      itemTokens.retainAll(queryTokens);
      score += itemTokens.size();
    }
    score /= shownitems.length;
    score /= queryTokens.size();
    return score;
  }

  public static String getItemStrings(WalmartQuery query, int[] items, Map<Integer, String> itemLookup) {
    StringBuilder buffer = new StringBuilder();
    for(int i = 0; i < items.length; ++i) {
      if (query.find(query.clickeditems, items[i]) >= 0){
        buffer.append(String.format("*%d: %s\n", i, itemLookup.get(items[i])));
      }
      else {
        buffer.append(String.format("%d: %s\n", i, itemLookup.get(items[i])));
      }
      
    }
    return buffer.toString();
  }

  @Override
  public int compareTo(WalmartQuery o) {
    return 0;
  }
}
