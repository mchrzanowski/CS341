import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.json.JSONArray;
import org.json.JSONObject;

public class UserItemRatingGenerator {

	public static class Map extends Mapper<LongWritable, Text, Text, Text> {
		private Text userItemPair = new Text();
		private Text valueTuple = new Text();

		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {    	
    	try {
    	    WalmartQuery walmartQuery = new WalmartQuery(new JSONObject(value.toString()));
    	    WalmartQuery.ResultItemRating[] resultItemRatings = walmartQuery.resultItemRatings;
            for(int i=0;i<resultItemRatings.length;i++) {
               String userItemPairStr = walmartQuery.visiorid+"#"+resultItemRatings[i].itemId;
   			   userItemPair.set(userItemPairStr);
   			   valueTuple.set(resultItemRatings[i].finalRating.getId()+"#"+walmartQuery.wmsessionid+"#"+walmartQuery.searchtimestamp);
   	           context.write(userItemPair, valueTuple);
            }    	   
    	}catch(Exception e){
    		System.out.println("[UserItemRatingGenerator] Error parsing JSON..."+e.getMessage());
    	}    	
      }
	}

	public static class Reduce extends Reducer<Text, Text, Text, NullWritable> {
		private NullWritable nullValue = NullWritable.get();
		private Text key = new Text();

		public void reduce(Text userItemPair, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {			
			String[] tokens=userItemPair.toString().split("#");			
			PairRatingRecord pairRatingRecord = RankingUtil.getPairRating(values);
			RankingUtil.writeRecord(context,  key, tokens[0],  tokens[1], pairRatingRecord.rating.getId(), 
									pairRatingRecord.searchtimestamp, pairRatingRecord.wmsessionid);
		}
	}

}
