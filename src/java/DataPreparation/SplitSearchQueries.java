import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.json.JSONArray;
import org.json.JSONObject;

public class SplitSearchQueries {

	public static class Map extends Mapper<LongWritable, Text, Text, Text> {
		private Text userId = new Text();
		private Text clickCount_QueryRecord = new Text();
		
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			try {
				JSONObject jsonObject = new JSONObject(value.toString());
				String visiorId = jsonObject.getString("visitorid");
				JSONArray clickeditems = jsonObject.getJSONArray("clickeditems");
				
				String searchTimeStamp = jsonObject.getString("searchtimestamp");
				
				if(searchTimeStamp == null) return;
				
				int totalClicks = 0;
				totalClicks += clickeditems.length();

				userId.set(visiorId);
				clickCount_QueryRecord.set(totalClicks + "#" + value.toString());
				
				if((visiorId == null)||(visiorId.trim().equals(""))) return;
				context.write(userId, clickCount_QueryRecord);
			} catch (Exception e) {
				System.out.println("Error parsing JSON..."+e.getMessage());
			}
		}
	}

	public static class Reduce extends Reducer<Text, Text, Text, NullWritable> {
		private Text queryRecord = new Text();
		private MultipleOutputs mos;
		private static float TRAINING_SPLIT_PERCENTAGE = 0.8f;
		private int CLICK_THRESHOLD ;
		
		private static String TRAINING_LOC;
		private static String TESTING_LOC;
		
		public void setup(Context context) {
			mos = new MultipleOutputs(context);
		}

		@SuppressWarnings("unchecked")
		public void reduce(Text userId, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {
			int totalClicks = 0;

			TRAINING_LOC = context.getConfiguration().get("TRAINING_LOC");
			TESTING_LOC = context.getConfiguration().get("TESTING_LOC");
			CLICK_THRESHOLD = Integer.parseInt(context.getConfiguration().get("CLICK_THRESHOLD"));
			
			List<String> queryRecords = new ArrayList<String>();
			for (Text line : values) {
				String[] valueTokens = line.toString().split("#");
				totalClicks += Integer.parseInt(valueTokens[0]);
				
				String tmpStr = "";
				for(int i=1;i<valueTokens.length;i++){
					tmpStr += valueTokens[i];
				}
				queryRecords.add(tmpStr);
			}
			
			System.out.println("Click threshold"+CLICK_THRESHOLD);
			if(totalClicks < CLICK_THRESHOLD) return;

			Collections.sort(queryRecords, new Comparator() {
				public int compare(Object obj1, Object obj2) {
					try {
						JSONObject queryObject1 = new JSONObject((String)obj1);
						long searchtimestampQueryObject1 = Long.parseLong(queryObject1.getString("searchtimestamp"));
						JSONObject queryObject2 = new JSONObject((String)obj2);
						long searchtimestampQueryObject2 = Long.parseLong(queryObject2.getString("searchtimestamp"));
						if(searchtimestampQueryObject1<searchtimestampQueryObject2) return -1;
						else if(searchtimestampQueryObject1 == searchtimestampQueryObject2) return 0;
						else return 1;
					}catch(Exception e) {
						System.out.println("Error while comparing:"+e.getMessage());
					}
					return -1;
				}				
			});
			
			for (int i = 0; i < Math.round(queryRecords.size()*TRAINING_SPLIT_PERCENTAGE); i++) {
				queryRecord.set(queryRecords.get(i));
				mos.write("training",queryRecord, NullWritable.get(), TRAINING_LOC);
			}

			for (int i = Math.round(queryRecords.size()*TRAINING_SPLIT_PERCENTAGE); i < queryRecords.size(); i++) {
				queryRecord.set(queryRecords.get(i));
				mos.write("testing", queryRecord, NullWritable.get(), TESTING_LOC);
			}
		}
		
		 public void cleanup(Context context) throws IOException {
			  try{
				  mos.close();
			  }catch(Exception e){}
	     }
	}

}
