package SVD;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

import org.apache.commons.math.linear.Array2DRowRealMatrix;
import org.apache.commons.math.linear.RealMatrix;
import org.apache.commons.math.linear.RealVector;

public class SVDImpl {
	public static void main(String[] args) {
		String whole_input_file = "/Users/sri/Desktop/341data/svd_input";
		String training_file = "/Users/sri/Desktop/341data/svd_training";
		String test_file = "/Users/sri/Desktop/341data/svd_test";
		
		final long TOP_RECOMMENDATIONS_COUNT = 200;
		
		HashMap<Long, Integer> itemsNewIdDict = new HashMap<Long, Integer>();
		HashMap<Long, Integer> usersNewIdDict = new HashMap<Long, Integer>();
		HashMap<Integer, Long> itemsReverseLookup = new HashMap<Integer, Long>();
		
		Map<Integer, Set> topkPredictionsForUsers = new HashMap<Integer,Set>();
		Map<Integer, Set> userItemClickedOrOrdered = new HashMap<Integer, Set>();
		
		TreeSet<ItemStats> allPopularItems = new TreeSet<ItemStats>();
		Set topPopularItems = new HashSet();
		
		try {
			String s;
			FileReader fr = new FileReader(whole_input_file);
			BufferedReader br = new BufferedReader(fr);

			Integer totalUserCount = new Integer(0);
			Integer totalItemCount = new Integer(0);
			Integer totalRecords = new Integer(0);
			
			Map<Integer, ItemStats> itemCount = new HashMap<Integer, ItemStats>();
			while ((s = br.readLine()) != null) {
				String[] tokens = s.split("\t");
				String token_userid = tokens[0]; // userid
				String token_itemid = tokens[1]; // itemid
				String token_label = tokens[2]; // label

				Integer new_item_id = itemsNewIdDict.get(new Long(token_itemid));
				if (new_item_id == null) {
					new_item_id = itemsNewIdDict.size();
					itemsNewIdDict.put(new Long(token_itemid), new_item_id);
					itemsReverseLookup.put(new_item_id,Long.parseLong(token_itemid));
				}

				Integer new_user_id = usersNewIdDict.get(new Long(token_userid));
				if (new_user_id == null) {
					new_user_id = usersNewIdDict.size();
					System.out.println("user size.."+new_user_id);
					usersNewIdDict.put(new Long(token_userid), new_user_id);
				}
				totalRecords++;
				
				ItemStats itemStats = (ItemStats)itemCount.get(new_item_id);
				if(itemStats == null)  { 
					itemStats = new ItemStats(new Long(new_item_id));
					itemCount.put(new_item_id, itemStats);
				}
				
				switch(Integer.parseInt(token_label)) {
					case 1: itemStats.totalShowedCount +=1; break;
					case 2: itemStats.totalClickedCount +=1; break;
					case 3: itemStats.totalOrderedCount +=1; break;
				}
				itemStats.totalCount += 1;
			}
			br.close();
			fr.close();

			allPopularItems.addAll(itemCount.values());
			
			System.out.println("*******************************");
			System.out.println("Top 5 popular items:");

			int index = 0;
			Iterator popularItemsIt = allPopularItems.iterator();
			while(popularItemsIt.hasNext()) {
				ItemStats itemStats = (ItemStats)popularItemsIt.next();
				System.out.println("       "+itemStats.itemId+":"+itemStats.totalCount +":"+itemStats.totalShowedCount
						                    +":"+itemStats.totalClickedCount+":"+itemStats.totalOrderedCount);
				index++;
				topPopularItems.add(itemStats.itemId);
				if(index>TOP_RECOMMENDATIONS_COUNT) break;
			}			
			
			System.out.println("*******************************");
			System.out.println(" - itemsize:" + itemsNewIdDict.size());
			System.out.println(" - usersize:" + usersNewIdDict.size());
			System.out.println(" - totalRecords:" + totalRecords);
			System.out.println("*******************************");
			
			int m = usersNewIdDict.size();
			int n = itemsNewIdDict.size();
			int MAX_ITERATIONS = 40;

	//		double[] lambdas = new double[] { 0.1, 0.01, 0.04, 0.07, 0.3, 0.5};
			double[] lambdas = new double[] { 0.1};

			
			for (int l = 0; l < lambdas.length; l++) {
				for (int k = 18; k < 19; k++) {
					double lambda = lambdas[l];
					double learning_rate = 0.03;
					double[][] PD = new double[m][k];
					double[][] QD = new double[n][k];
					for (int i = 0; i < m; i++) {
						for (int j = 0; j < k; j++) {
							PD[i][j] = Math.random() * Math.pow(3.0 / k, 0.5);
						}
					}
					for (int i = 0; i < n; i++) {
						for (int j = 0; j < k; j++) {
							QD[i][j] = Math.random() * Math.pow(3.0 / k, 0.5);
						}
					}

					RealMatrix P = new Array2DRowRealMatrix(PD);
					RealMatrix Q = new Array2DRowRealMatrix(QD);

					for (int i = 0; i < MAX_ITERATIONS; i++) {
						fr = new FileReader(training_file);
						br = new BufferedReader(fr);

						int totalprocessed = 0;
						double error = 0;
						double PNorm = Math.pow(P.getNorm(), 2);
						double QNorm = Math.pow(Q.getNorm(), 2);
						error = error + lambda * (PNorm + QNorm);
						while ((s = br.readLine()) != null) {
							String[] tokens = s.split("\t");
							String token_userid = tokens[0]; // userid
							String token_itemid = tokens[1]; // itemid
							String token_label = tokens[2]; // label

							Integer new_item_id = itemsNewIdDict.get(new Long(token_itemid));
							Integer new_user_id = usersNewIdDict.get(new Long(token_userid));

							int label = Integer.parseInt(token_label);
							
							double episilon = label - Q.getRowVector(new_item_id).dotProduct(P.getRowVector(new_user_id));

							if(label == 1) { //clicked
								learning_rate = 0.01;
							}
							else if(label == 2) { //placed in cart
								learning_rate = 0.01;								
							}
							else if(label == 3) { //ordered
								learning_rate = 0.01;								
							}
							
							if((label == 2)||(label==3)) {
								Set userItemSet = userItemClickedOrOrdered.get(new_user_id);
								if(userItemSet == null) {
									userItemSet = new HashSet();
									userItemSet.add(new_item_id);
								}
								userItemClickedOrOrdered.put(new_user_id, userItemSet);
							}
							
							RealVector updatedQ = Q.getRowVector(new_item_id).add((P.getRowVector(new_user_id).mapMultiplyToSelf(episilon)
																			.subtract(Q.getRowVector(new_item_id).mapMultiplyToSelf(lambda))
																			.mapMultiplyToSelf(learning_rate)));
							RealVector updatedP = P.getRowVector(new_user_id).add((Q.getRowVector(new_item_id).mapMultiplyToSelf(episilon)
																			.subtract(P.getRowVector(new_user_id).mapMultiplyToSelf(lambda))
																			.mapMultiplyToSelf(learning_rate)));
							error = error + Math.pow(episilon, 2);
							Q.setRowVector(new_item_id, updatedQ);
							P.setRowVector(new_user_id, updatedP);
							totalprocessed++;

							if (totalprocessed % 1000000 == 0)
								System.out.println("totalprocessed:"
										+ totalprocessed + ";userid:"
										+ token_userid + ";itemid:"
										+ token_itemid + ";label:"
										+ token_label);
						}

						br.close();
						fr.close();					
					}

					//Now calculating top k item recommendations for each user
					
					for(int i=0;i<P.getRowDimension();i++) {
						RealMatrix userPredictedRatings = P.getRowMatrix(i).multiply(Q.transpose());
						Set topKitems = topkPredictionsForUsers.get(i);
						if(topKitems == null) {
							topKitems = new HashSet();
							topkPredictionsForUsers.put(i,  topKitems);
						}

						Set clickedItems = new HashSet();
						Set showedItems = new HashSet();
						
						int orderedcount = 0;
						int clickedcount = 0;
						int showncount = 0;
						
						Set userItemSet = userItemClickedOrOrdered.get(i);
						for(int j=0;j<userPredictedRatings.getColumnDimension();j++) {
							if(userItemSet != null) {
								if(userItemSet.contains(j)) continue;
							}
                            
							long rating = Math.round(userPredictedRatings.getEntry(0,j));
							if(rating == 3) {
								topKitems.add(j);
								orderedcount++;
							}
							else if(rating == 2) {
								clickedItems.add(j);
							}
							else if(rating == 1) {
								showedItems.add(j);
							} 
							
							if(topKitems.size()>TOP_RECOMMENDATIONS_COUNT) break;
						}
		
						Iterator clickedIt = clickedItems.iterator();
						while(clickedIt.hasNext() && topKitems.size() < TOP_RECOMMENDATIONS_COUNT) {
							topKitems.add(clickedIt.next());
							clickedcount++;
						}
							
						Iterator shownIt = showedItems.iterator();
						while(shownIt.hasNext() && topKitems.size() < TOP_RECOMMENDATIONS_COUNT) {
							topKitems.add(shownIt.next());
							showncount++;
						}
						System.out.println("**Computing for user:"+i+":"+showncount+":"+clickedcount+":"+orderedcount);
					}
					
					//Running against test data
					double test_error = 0;
					fr = new FileReader(test_file);
					br = new BufferedReader(fr);

					int totalTestRecords = 0;
					int totalCorrectlyClassified = 0;
							
					HashMap labelTotalCounts = new HashMap();
					HashMap labelTruePositiveCounts = new HashMap();
					HashMap labelFalsePositiveCounts = new HashMap();
					 
					int tpwrtMostPopular = 0;
					int tpwrtUserRecommendations = 0;
					int alreadyClickedOrOrdered = 0;
					
					while ((s = br.readLine()) != null) {
						String[] tokens = s.split("\t");
						String token_userid = tokens[0]; // userid
						String token_itemid = tokens[1]; // itemid
						String token_label = tokens[2]; // label

						int new_item_id = itemsNewIdDict.get(new Long(token_itemid));
						int new_user_id = usersNewIdDict.get(new Long(token_userid));
						
						test_error = test_error+ Math.pow((Integer.parseInt(token_label) - Q.getRowVector(new_item_id).dotProduct(P.getRowVector(new_user_id))),2);
						
						totalTestRecords++;
						
						if (token_label.equals("2") || token_label.equals("3")) {
							if (topPopularItems.contains(new_item_id)) {
								tpwrtMostPopular++;
							}

							Set userItemRecommendations = topkPredictionsForUsers.get(new_user_id);
							if (userItemRecommendations.contains(new_item_id)) {
								tpwrtUserRecommendations++;
							}
							
							Set items = userItemClickedOrOrdered.get(new_user_id);
							
							if((items!=null)&&(items.contains(new_item_id))) {
								alreadyClickedOrOrdered++;
							}
						}

						if((Integer.parseInt(token_label) - Math.round(Q.getRowVector(new_item_id).dotProduct(P.getRowVector(new_user_id)))) == 0) {
							totalCorrectlyClassified++;	
							
							Integer correctlyPredicatedLabelCount = (Integer)labelTruePositiveCounts.get(token_label);
							if(correctlyPredicatedLabelCount == null) {
								labelTruePositiveCounts.put(token_label, new Integer(1));
							}
							else {
								labelTruePositiveCounts.put(token_label, ++correctlyPredicatedLabelCount);
							}	
						}
						else {
							Integer predictedLabel=(int)Math.round(Q.getRowVector(new_item_id).dotProduct(P.getRowVector(new_user_id)));
							Integer incorrectlyPredicatedLabelCount = (Integer)labelFalsePositiveCounts.get(predictedLabel.toString());
							if(incorrectlyPredicatedLabelCount == null) {
								labelFalsePositiveCounts.put(predictedLabel.toString(), new Integer(1));
							}
							else {
								labelFalsePositiveCounts.put(predictedLabel.toString(), ++incorrectlyPredicatedLabelCount);
							}
						}
						
						Integer labelCount = (Integer)labelTotalCounts.get(token_label);
						if(labelCount == null) {
							labelTotalCounts.put(token_label, new Integer(1));
						}
						else {
							labelTotalCounts.put(token_label, ++labelCount);
						}
					}
					System.out.print(lambda + ","+k+"," + test_error+","+totalTestRecords+","+totalCorrectlyClassified+","+(((double)totalCorrectlyClassified)/totalTestRecords));
					Iterator it = labelTotalCounts.keySet().iterator(); 
					while(it.hasNext()){
						String label = (String)it.next();
						Integer labelcount = (Integer)labelTotalCounts.get(label);
						Integer labelTruePositivesCount = (Integer)labelTruePositiveCounts.get(label);
						if(labelTruePositivesCount == null) {
							labelTruePositivesCount = 0;
						}
						
						Integer labelFalsePositivesCount = (Integer)labelFalsePositiveCounts.get(label);
						if(labelFalsePositivesCount == null) {
							labelFalsePositivesCount = 0;
						}
						System.out.print(",\n"+label+"=(total:"+labelcount+";tp:"+labelTruePositivesCount+";fp:"+labelFalsePositivesCount+";recall:"+((double)labelTruePositivesCount/labelcount)+";precision:"+((double)labelTruePositivesCount/(labelFalsePositivesCount+labelTruePositivesCount))+")");
					}
					System.out.println("");
					
                    System.out.println("Total accurate predictions based on most popular:"+tpwrtMostPopular);
                    System.out.println("Total accurate predictions based on SVD top k User recomemndations:"+tpwrtUserRecommendations);
                    System.out.println("Total already clicked items:"+alreadyClickedOrOrdered);
				}
			}
		} catch (Exception e) {
			System.out.println("Error processing..." + e.getMessage());
			e.printStackTrace();
		}
	}
}
