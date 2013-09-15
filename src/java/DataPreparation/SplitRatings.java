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

public class SplitRatings {

	public static class Map extends Mapper<LongWritable, Text, Text, Text> {
		private Text outputkey = new Text();
		private Text outputvalue = new Text();
		
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			try {
				String line = value.toString();
				String[] tokens = line.split("\t");
				
				outputkey.set(tokens[0]);
				outputvalue.set(tokens[1]+"#"+tokens[2]+"#"+tokens[3]+"#"+tokens[4]);
				context.write(outputkey, outputvalue);
			} catch (Exception e) {
				System.out.println("Error parsing JSON..."+e.getMessage());
			}
		}
	}

	public static class Reduce extends Reducer<Text, Text, Text, NullWritable> {
		private Text outputKey = new Text();
		private MultipleOutputs mos;
		private static float TRAINING_SPLIT_PERCENTAGE = 0.8f;
		private static float VALIDATION_SPLIT_PERCENTAGE = 0.1f;
		private static String TRAINING_LOC;
		private static String RECENT_2_3_LOC;
		private static String RECENT_1_3_LOC;
		private static String TESTING_LOC;
		private static String VALIDATION_LOC;
		
		public void setup(Context context) {
			mos = new MultipleOutputs(context);
			TRAINING_LOC = context.getConfiguration().get("TRAINING_LOC");
			TESTING_LOC = context.getConfiguration().get("TESTING_LOC");
			VALIDATION_LOC = context.getConfiguration().get("VALIDATION_LOC");
			RECENT_2_3_LOC = context.getConfiguration().get("RECENT_2_3_LOC");
			RECENT_1_3_LOC = context.getConfiguration().get("RECENT_1_3_LOC");
		}

		public void reduce(Text key, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {			
			List<String> ratingRecords = new ArrayList<String>();
			for (Text line : values) {
				ratingRecords.add(line.toString());
			}
			
			Collections.sort(ratingRecords, new Comparator() {
				public int compare(Object obj1, Object obj2) {
					String[] ratingrec1_tokens = ((String)obj1).split("#");
					String[] ratingrec2_tokens = ((String)obj2).split("#");
					
					if(Long.parseLong(ratingrec1_tokens[2]) < Long.parseLong(ratingrec2_tokens[2])) {
						return -1;
					}
					else if(Long.parseLong(ratingrec1_tokens[2]) == Long.parseLong(ratingrec2_tokens[2])) {
						return 0;
					}
					else return 1;
				}				
			});			
			
			for (int i = (int)Math.floor(ratingRecords.size()*TRAINING_SPLIT_PERCENTAGE*1.0/3.0); i < Math.floor(ratingRecords.size()*TRAINING_SPLIT_PERCENTAGE); i++) {
				outputKey.set(key+"\t"+ratingRecords.get(i).replace("#", "\t"));
				mos.write("recent23",outputKey, NullWritable.get(), RECENT_2_3_LOC);
			}
			
			for (int i = (int)Math.floor(ratingRecords.size()*TRAINING_SPLIT_PERCENTAGE*2.0/3.0); i < Math.floor(ratingRecords.size()*TRAINING_SPLIT_PERCENTAGE); i++) {
				outputKey.set(key+"\t"+ratingRecords.get(i).replace("#", "\t"));
				mos.write("recent13",outputKey, NullWritable.get(), RECENT_1_3_LOC);
			}
			
			for (int i = 0; i < Math.round(ratingRecords.size()*TRAINING_SPLIT_PERCENTAGE); i++) {
				outputKey.set(key+"\t"+ratingRecords.get(i).replace("#", "\t"));
				mos.write("training",outputKey, NullWritable.get(), TRAINING_LOC);
			}

			for (int i = (int) Math.round(ratingRecords.size()*TRAINING_SPLIT_PERCENTAGE); i < Math.round(ratingRecords.size()*(TRAINING_SPLIT_PERCENTAGE+VALIDATION_SPLIT_PERCENTAGE)); i++) {
				outputKey.set(key+"\t"+ratingRecords.get(i).replace("#", "\t"));
				mos.write("validation",outputKey, NullWritable.get(), VALIDATION_LOC);
			}
			
			for (int i = (int)Math.round(ratingRecords.size()*(TRAINING_SPLIT_PERCENTAGE+VALIDATION_SPLIT_PERCENTAGE)); i < ratingRecords.size(); i++) {
				outputKey.set(key+"\t"+ratingRecords.get(i).replace("#", "\t"));
				mos.write("testing",outputKey, NullWritable.get(), TESTING_LOC);
			}
		}
		
		 public void cleanup(Context context) throws IOException {
			  try{
				  mos.close();
			  }catch(Exception e){}
	     }
	}

}
