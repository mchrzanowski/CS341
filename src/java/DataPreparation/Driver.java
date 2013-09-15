import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;


public class Driver {
	public static void main(String[] args) throws Exception {
		Constants.OUTPUT_BASE_PATH = args[1]; 
		
		int CLICK_THRESHOLD = 50;
		if(args.length==3){
			CLICK_THRESHOLD = Integer.parseInt(args[2]);
		}
		
		Configuration conf = new Configuration();	        
	    
		//1. Identifying users with clicks>'x' clicks and spliting into train and test files.
	    conf.set("TRAINING_LOC", Constants.OUTPUT_BASE_PATH+Constants.QUERY_TRAINING_LOC);
	    conf.set("TESTING_LOC", Constants.OUTPUT_BASE_PATH+Constants.QUERY_TESTING_LOC);
	    conf.set("CLICK_THRESHOLD",""+CLICK_THRESHOLD);
	    Job job = new Job(conf, "Driver");
	    job.setJarByClass(SplitSearchQueries.class);
	    job.setOutputKeyClass(Text.class);
	    job.setOutputValueClass(NullWritable.class);
	        
	    job.setMapOutputKeyClass(Text.class);
	    job.setMapOutputValueClass(Text.class);
        
	    job.setMapperClass(SplitSearchQueries.Map.class);
	    job.setReducerClass(SplitSearchQueries.Reduce.class);
	        
	    job.setInputFormatClass(TextInputFormat.class);
	    job.setOutputFormatClass(TextOutputFormat.class);
	        
	    FileInputFormat.addInputPath(job, new Path(args[0]));
	    FileOutputFormat.setOutputPath(job, new Path(args[1]));
	        
	    MultipleOutputs.addNamedOutput(job, "training", TextOutputFormat.class, Text.class, NullWritable.class);
	    MultipleOutputs.addNamedOutput(job, "testing", TextOutputFormat.class, Text.class, NullWritable.class);
	    job.waitForCompletion(true);
	      
	    //2. Doing item rank rating (NO Duplicates)
	    conf.set("DUPLICATES_ALLOWED", "false");
	    job = new Job(conf, "ItemRank");
	    job.setJarByClass(ItemRankRatingGenerator.class);
	    job.setOutputKeyClass(Text.class);
	    job.setOutputValueClass(NullWritable.class);
	        
	    job.setMapOutputKeyClass(Text.class);
	    job.setMapOutputValueClass(Text.class);
        
	    job.setMapperClass(ItemRankRatingGenerator.Map.class);
	    job.setReducerClass(ItemRankRatingGenerator.Reduce.class);
	        
	    job.setInputFormatClass(TextInputFormat.class);
	    job.setOutputFormatClass(TextOutputFormat.class);
	        
	    FileInputFormat.addInputPath(job, new Path(Constants.OUTPUT_BASE_PATH+Constants.QUERY_TRAINING_LOC));
	    FileOutputFormat.setOutputPath(job, new Path(Constants.OUTPUT_BASE_PATH+Constants.ITEM_RANK_NON_DUPLICATES_LOC));	        
	    job.waitForCompletion(true);
	    
	    //3. Doing user item rating  (NO Duplicates)
	    conf.set("DUPLICATES_ALLOWED", "false");
	    job = new Job(conf, "UserItem");
	    job.setJarByClass(UserItemRatingGenerator.class);
	    job.setOutputKeyClass(Text.class);
	    job.setOutputValueClass(NullWritable.class);
	        
	    job.setMapOutputKeyClass(Text.class);
	    job.setMapOutputValueClass(Text.class);
        
	    job.setMapperClass(UserItemRatingGenerator.Map.class);
	    job.setReducerClass(UserItemRatingGenerator.Reduce.class);
	        
	    job.setInputFormatClass(TextInputFormat.class);
	    job.setOutputFormatClass(TextOutputFormat.class);
	        
	    FileInputFormat.addInputPath(job, new Path(Constants.OUTPUT_BASE_PATH+Constants.QUERY_TRAINING_LOC));
	    FileOutputFormat.setOutputPath(job, new Path(Constants.OUTPUT_BASE_PATH+Constants.USER_ITEM_NON_DUPLICATES_LOC));
	    job.waitForCompletion(true);
	    	    	    
	   //4, Doing user rank rating  (NO Duplicates)
	    conf.set("DUPLICATES_ALLOWED", "false");
	    job = new Job(conf, "UserRank");
	    job.setJarByClass(UserRankRatingGenerator.class);
	    job.setOutputKeyClass(Text.class);
	    job.setOutputValueClass(NullWritable.class);
	        
	    job.setMapOutputKeyClass(Text.class);
	    job.setMapOutputValueClass(Text.class);
        
	    job.setMapperClass(UserRankRatingGenerator.Map.class);
	    job.setReducerClass(UserRankRatingGenerator.Reduce.class);
	        
	    job.setInputFormatClass(TextInputFormat.class);
	    job.setOutputFormatClass(TextOutputFormat.class);
	        
	    FileInputFormat.addInputPath(job, new Path(Constants.OUTPUT_BASE_PATH+Constants.QUERY_TRAINING_LOC));
	    FileOutputFormat.setOutputPath(job, new Path(Constants.OUTPUT_BASE_PATH+Constants.USER_RANK_NON_DUPLICATES_LOC));
	    
	    job.waitForCompletion(true);
	    
	    //5. Spliting ItemRank records  (NO Duplicates)
	    conf = new Configuration();	
	    conf.set("DUPLICATES_ALLOWED", "false");
	    conf.set("TRAINING_LOC", Constants.OUTPUT_BASE_PATH+Constants.ITEM_RANK_TRAINING_NON_DUPLICATES_LOC);
	    conf.set("VALIDATION_LOC", Constants.OUTPUT_BASE_PATH+Constants.ITEM_RANK_VALIDATION_NON_DUPLICATES_LOC);
	    conf.set("TESTING_LOC", Constants.OUTPUT_BASE_PATH+Constants.ITEM_RANK_TESTING_NON_DUPLICATES_LOC);
	    conf.set("RECENT_2_3_LOC", Constants.OUTPUT_BASE_PATH+Constants.ITEM_RANK_TRAINING_RECENT_2_3_NON_DUPLICATES_LOC);
	    conf.set("RECENT_1_3_LOC", Constants.OUTPUT_BASE_PATH+Constants.ITEM_RANK_TRAINING_RECENT_1_3_NON_DUPLICATES_LOC);
	    job = new Job(conf, "ItemRankSplit");
	    job.setJarByClass(SplitRatings.class);
	    job.setOutputKeyClass(Text.class);
	    job.setOutputValueClass(NullWritable.class);
	        
	    job.setMapOutputKeyClass(Text.class);
	    job.setMapOutputValueClass(Text.class);
        
	    job.setMapperClass(SplitRatings.Map.class);
	    job.setReducerClass(SplitRatings.Reduce.class);
	        
	    job.setInputFormatClass(TextInputFormat.class);
	    job.setOutputFormatClass(TextOutputFormat.class);
	        
	    FileInputFormat.addInputPath(job, new Path(Constants.OUTPUT_BASE_PATH+Constants.ITEM_RANK_NON_DUPLICATES_LOC));
	    FileOutputFormat.setOutputPath(job, new Path(args[1]+"/junk1"));
	        
	    MultipleOutputs.addNamedOutput(job, "training", TextOutputFormat.class, Text.class, NullWritable.class);
	    MultipleOutputs.addNamedOutput(job, "validation", TextOutputFormat.class, Text.class, NullWritable.class);
	    MultipleOutputs.addNamedOutput(job, "testing", TextOutputFormat.class, Text.class, NullWritable.class);
	    MultipleOutputs.addNamedOutput(job, "recent23", TextOutputFormat.class, Text.class, NullWritable.class);
	    MultipleOutputs.addNamedOutput(job, "recent13", TextOutputFormat.class, Text.class, NullWritable.class);
	    job.waitForCompletion(true);
	  
	   //6. Spliting UserRank records (NO Duplicates)
	    conf = new Configuration();	
	    conf.set("DUPLICATES_ALLOWED", "false");
	    conf.set("TRAINING_LOC", Constants.OUTPUT_BASE_PATH+Constants.USER_RANK_TRAINING_NON_DUPLICATES_LOC);
	    conf.set("VALIDATION_LOC", Constants.OUTPUT_BASE_PATH+Constants.USER_RANK_VALIDATION_NON_DUPLICATES_LOC);
	    conf.set("TESTING_LOC", Constants.OUTPUT_BASE_PATH+Constants.USER_RANK_TESTING_NON_DUPLICATES_LOC);
	    conf.set("RECENT_2_3_LOC", Constants.OUTPUT_BASE_PATH+Constants.USER_RANK_TRAINING_RECENT_2_3_NON_DUPLICATES_LOC);
	    conf.set("RECENT_1_3_LOC", Constants.OUTPUT_BASE_PATH+Constants.USER_RANK_TRAINING_RECENT_1_3_NON_DUPLICATES_LOC);	    
	    job = new Job(conf, "UserRankSplit");
	    job.setJarByClass(SplitRatings.class);
	    job.setOutputKeyClass(Text.class);
	    job.setOutputValueClass(NullWritable.class);
	        
	    job.setMapOutputKeyClass(Text.class);
	    job.setMapOutputValueClass(Text.class);
        
	    job.setMapperClass(SplitRatings.Map.class);
	    job.setReducerClass(SplitRatings.Reduce.class);
	        
	    job.setInputFormatClass(TextInputFormat.class);
	    job.setOutputFormatClass(TextOutputFormat.class);
	        
	    FileInputFormat.addInputPath(job, new Path(Constants.OUTPUT_BASE_PATH+Constants.USER_RANK_NON_DUPLICATES_LOC));
	    FileOutputFormat.setOutputPath(job, new Path(args[1]+"/junk2"));
	        
	    MultipleOutputs.addNamedOutput(job, "training", TextOutputFormat.class, Text.class, NullWritable.class);
	    MultipleOutputs.addNamedOutput(job, "validation", TextOutputFormat.class, Text.class, NullWritable.class);
	    MultipleOutputs.addNamedOutput(job, "testing", TextOutputFormat.class, Text.class, NullWritable.class);
	    MultipleOutputs.addNamedOutput(job, "recent23", TextOutputFormat.class, Text.class, NullWritable.class);
	    MultipleOutputs.addNamedOutput(job, "recent13", TextOutputFormat.class, Text.class, NullWritable.class);
	    job.waitForCompletion(true);

		//7. Spliting UserItem records  (NO Duplicates)
	    conf = new Configuration();	
	    conf.set("DUPLICATES_ALLOWED", "false");
	    conf.set("TRAINING_LOC", Constants.OUTPUT_BASE_PATH+Constants.USER_ITEM_TRAINING_NON_DUPLICATES_LOC);
	    conf.set("VALIDATION_LOC", Constants.OUTPUT_BASE_PATH+Constants.USER_ITEM_VALIDATION_NON_DUPLICATES_LOC);
	    conf.set("TESTING_LOC", Constants.OUTPUT_BASE_PATH+Constants.USER_ITEM_TESTING_NON_DUPLICATES_LOC);
	    conf.set("RECENT_2_3_LOC", Constants.OUTPUT_BASE_PATH+Constants.USER_ITEM_TRAINING_RECENT_2_3_NON_DUPLICATES_LOC);
	    conf.set("RECENT_1_3_LOC", Constants.OUTPUT_BASE_PATH+Constants.USER_ITEM_TRAINING_RECENT_1_3_NON_DUPLICATES_LOC);
	    job = new Job(conf, "UserItemSplit");
	    job.setJarByClass(SplitRatings.class);
	    job.setOutputKeyClass(Text.class);
	    job.setOutputValueClass(NullWritable.class);
	        
	    job.setMapOutputKeyClass(Text.class);
	    job.setMapOutputValueClass(Text.class);
        
	    job.setMapperClass(SplitRatings.Map.class);
	    job.setReducerClass(SplitRatings.Reduce.class);
	        
	    job.setInputFormatClass(TextInputFormat.class);
	    job.setOutputFormatClass(TextOutputFormat.class);
	        
	    FileInputFormat.addInputPath(job, new Path(Constants.OUTPUT_BASE_PATH+Constants.USER_ITEM_NON_DUPLICATES_LOC));
	    FileOutputFormat.setOutputPath(job, new Path(args[1]+"/junk3"));
	        
	    MultipleOutputs.addNamedOutput(job, "training", TextOutputFormat.class, Text.class, NullWritable.class);
	    MultipleOutputs.addNamedOutput(job, "validation", TextOutputFormat.class, Text.class, NullWritable.class);
	    MultipleOutputs.addNamedOutput(job, "testing", TextOutputFormat.class, Text.class, NullWritable.class);
	    MultipleOutputs.addNamedOutput(job, "recent23", TextOutputFormat.class, Text.class, NullWritable.class);
	    MultipleOutputs.addNamedOutput(job, "recent13", TextOutputFormat.class, Text.class, NullWritable.class);
	    job.waitForCompletion(true);
	    
	    //8. Doing item rank rating  (Duplicates)
	    conf = new Configuration();	
	    conf.set("DUPLICATES_ALLOWED", "true");
	    job = new Job(conf, "ItemRank2");
	    job.setJarByClass(ItemRankRatingGenerator.class);
	    job.setOutputKeyClass(Text.class);
	    job.setOutputValueClass(NullWritable.class);
	        
	    job.setMapOutputKeyClass(Text.class);
	    job.setMapOutputValueClass(Text.class);
        
	    job.setMapperClass(ItemRankRatingGenerator.Map.class);
	    job.setReducerClass(ItemRankRatingGenerator.Reduce.class);
	        
	    job.setInputFormatClass(TextInputFormat.class);
	    job.setOutputFormatClass(TextOutputFormat.class);
	        
	    FileInputFormat.addInputPath(job, new Path(Constants.OUTPUT_BASE_PATH+Constants.QUERY_TRAINING_LOC));
	    FileOutputFormat.setOutputPath(job, new Path(Constants.OUTPUT_BASE_PATH+Constants.ITEM_RANK_DUPLICATES_LOC));	        
	    job.waitForCompletion(true);
	    
	    //9. Doing user item rating   (Duplicates)
	    conf.set("DUPLICATES_ALLOWED", "true");
	    job = new Job(conf, "UserItem2");
	    job.setJarByClass(UserItemRatingGenerator.class);
	    job.setOutputKeyClass(Text.class);
	    job.setOutputValueClass(NullWritable.class);
	        
	    job.setMapOutputKeyClass(Text.class);
	    job.setMapOutputValueClass(Text.class);
        
	    job.setMapperClass(UserItemRatingGenerator.Map.class);
	    job.setReducerClass(UserItemRatingGenerator.Reduce.class);
	        
	    job.setInputFormatClass(TextInputFormat.class);
	    job.setOutputFormatClass(TextOutputFormat.class);
	        
	    FileInputFormat.addInputPath(job, new Path(Constants.OUTPUT_BASE_PATH+Constants.QUERY_TRAINING_LOC));
	    FileOutputFormat.setOutputPath(job, new Path(Constants.OUTPUT_BASE_PATH+Constants.USER_ITEM_DUPLICATES_LOC));
	    job.waitForCompletion(true);
	    	    
	    
	   //10 Doing user rank rating   (Duplicates)
	    conf.set("DUPLICATES_ALLOWED", "true");
	    job = new Job(conf, "UserRank2");
	    job.setJarByClass(UserRankRatingGenerator.class);
	    job.setOutputKeyClass(Text.class);
	    job.setOutputValueClass(NullWritable.class);
	        
	    job.setMapOutputKeyClass(Text.class);
	    job.setMapOutputValueClass(Text.class);
        
	    job.setMapperClass(UserRankRatingGenerator.Map.class);
	    job.setReducerClass(UserRankRatingGenerator.Reduce.class);
	        
	    job.setInputFormatClass(TextInputFormat.class);
	    job.setOutputFormatClass(TextOutputFormat.class);
	        
	    FileInputFormat.addInputPath(job, new Path(Constants.OUTPUT_BASE_PATH+Constants.QUERY_TRAINING_LOC));
	    FileOutputFormat.setOutputPath(job, new Path(Constants.OUTPUT_BASE_PATH+Constants.USER_RANK_DUPLICATES_LOC));
	    
	    job.waitForCompletion(true);
	    
	    
	    //11. Spliting ItemRank records   (Duplicates)
	    conf = new Configuration();	
	    conf.set("DUPLICATES_ALLOWED", "true");
	    conf.set("TRAINING_LOC", Constants.OUTPUT_BASE_PATH+Constants.ITEM_RANK_TRAINING_DUPLICATES_LOC);
	    conf.set("VALIDATION_LOC", Constants.OUTPUT_BASE_PATH+Constants.ITEM_RANK_VALIDATION_DUPLICATES_LOC);
	    conf.set("TESTING_LOC", Constants.OUTPUT_BASE_PATH+Constants.ITEM_RANK_TESTING_DUPLICATES_LOC);
	    conf.set("RECENT_2_3_LOC", Constants.OUTPUT_BASE_PATH+Constants.ITEM_RANK_TRAINING_RECENT_2_3_DUPLICATES_LOC);
	    conf.set("RECENT_1_3_LOC", Constants.OUTPUT_BASE_PATH+Constants.ITEM_RANK_TRAINING_RECENT_1_3_DUPLICATES_LOC);
	    job = new Job(conf, "ItemRankSplit2");
	    job.setJarByClass(SplitRatings.class);
	    job.setOutputKeyClass(Text.class);
	    job.setOutputValueClass(NullWritable.class);
	        
	    job.setMapOutputKeyClass(Text.class);
	    job.setMapOutputValueClass(Text.class);
        
	    job.setMapperClass(SplitRatings.Map.class);
	    job.setReducerClass(SplitRatings.Reduce.class);
	        
	    job.setInputFormatClass(TextInputFormat.class);
	    job.setOutputFormatClass(TextOutputFormat.class);
	        
	    FileInputFormat.addInputPath(job, new Path(Constants.OUTPUT_BASE_PATH+Constants.ITEM_RANK_DUPLICATES_LOC));
	    FileOutputFormat.setOutputPath(job, new Path(args[1]+"/junk4"));
	        
	    MultipleOutputs.addNamedOutput(job, "training", TextOutputFormat.class, Text.class, NullWritable.class);
	    MultipleOutputs.addNamedOutput(job, "validation", TextOutputFormat.class, Text.class, NullWritable.class);
	    MultipleOutputs.addNamedOutput(job, "testing", TextOutputFormat.class, Text.class, NullWritable.class);
	    MultipleOutputs.addNamedOutput(job, "recent23", TextOutputFormat.class, Text.class, NullWritable.class);
	    MultipleOutputs.addNamedOutput(job, "recent13", TextOutputFormat.class, Text.class, NullWritable.class);
	    job.waitForCompletion(true);
	  
	   //12. Spliting UserRank records   (Duplicates)
	    conf = new Configuration();	
	    conf.set("DUPLICATES_ALLOWED", "true");
	    conf.set("TRAINING_LOC", Constants.OUTPUT_BASE_PATH+Constants.USER_RANK_TRAINING_DUPLICATES_LOC);
	    conf.set("VALIDATION_LOC", Constants.OUTPUT_BASE_PATH+Constants.USER_RANK_VALIDATION_DUPLICATES_LOC);
	    conf.set("TESTING_LOC", Constants.OUTPUT_BASE_PATH+Constants.USER_RANK_TESTING_DUPLICATES_LOC);
	    conf.set("RECENT_2_3_LOC", Constants.OUTPUT_BASE_PATH+Constants.USER_RANK_TRAINING_RECENT_2_3_DUPLICATES_LOC);
	    conf.set("RECENT_1_3_LOC", Constants.OUTPUT_BASE_PATH+Constants.USER_RANK_TRAINING_RECENT_1_3_DUPLICATES_LOC);
	    job = new Job(conf, "UserRankSplit2");
	    job.setJarByClass(SplitRatings.class);
	    job.setOutputKeyClass(Text.class);
	    job.setOutputValueClass(NullWritable.class);
	        
	    job.setMapOutputKeyClass(Text.class);
	    job.setMapOutputValueClass(Text.class);
        
	    job.setMapperClass(SplitRatings.Map.class);
	    job.setReducerClass(SplitRatings.Reduce.class);
	        
	    job.setInputFormatClass(TextInputFormat.class);
	    job.setOutputFormatClass(TextOutputFormat.class);
	        
	    FileInputFormat.addInputPath(job, new Path(Constants.OUTPUT_BASE_PATH+Constants.USER_RANK_DUPLICATES_LOC));
	    FileOutputFormat.setOutputPath(job, new Path(args[1]+"/junk5"));
	        
	    MultipleOutputs.addNamedOutput(job, "training", TextOutputFormat.class, Text.class, NullWritable.class);
	    MultipleOutputs.addNamedOutput(job, "validation", TextOutputFormat.class, Text.class, NullWritable.class);
	    MultipleOutputs.addNamedOutput(job, "testing", TextOutputFormat.class, Text.class, NullWritable.class);
	    MultipleOutputs.addNamedOutput(job, "recent23", TextOutputFormat.class, Text.class, NullWritable.class);
	    MultipleOutputs.addNamedOutput(job, "recent13", TextOutputFormat.class, Text.class, NullWritable.class);
	    job.waitForCompletion(true);

		//13. Spliting UserItem records  (Duplicates)
	    conf = new Configuration();	
	    conf.set("DUPLICATES_ALLOWED", "true");
	    conf.set("TRAINING_LOC", Constants.OUTPUT_BASE_PATH+Constants.USER_ITEM_TRAINING_DUPLICATES_LOC);
	    conf.set("VALIDATION_LOC", Constants.OUTPUT_BASE_PATH+Constants.USER_ITEM_VALIDATION_DUPLICATES_LOC);
	    conf.set("TESTING_LOC", Constants.OUTPUT_BASE_PATH+Constants.USER_ITEM_TESTING_DUPLICATES_LOC);
	    conf.set("RECENT_2_3_LOC", Constants.OUTPUT_BASE_PATH+Constants.USER_ITEM_TRAINING_RECENT_2_3_DUPLICATES_LOC);
	    conf.set("RECENT_1_3_LOC", Constants.OUTPUT_BASE_PATH+Constants.USER_ITEM_TRAINING_RECENT_1_3_DUPLICATES_LOC);
	    job = new Job(conf, "UserItemSplit2");
	    job.setJarByClass(SplitRatings.class);
	    job.setOutputKeyClass(Text.class);
	    job.setOutputValueClass(NullWritable.class);
	        
	    job.setMapOutputKeyClass(Text.class);
	    job.setMapOutputValueClass(Text.class);
        
	    job.setMapperClass(SplitRatings.Map.class);
	    job.setReducerClass(SplitRatings.Reduce.class);
	        
	    job.setInputFormatClass(TextInputFormat.class);
	    job.setOutputFormatClass(TextOutputFormat.class);
	        
	    FileInputFormat.addInputPath(job, new Path(Constants.OUTPUT_BASE_PATH+Constants.USER_ITEM_DUPLICATES_LOC));
	    FileOutputFormat.setOutputPath(job, new Path(args[1]+"/junk6"));
	        
	    MultipleOutputs.addNamedOutput(job, "training", TextOutputFormat.class, Text.class, NullWritable.class);
	    MultipleOutputs.addNamedOutput(job, "validation", TextOutputFormat.class, Text.class, NullWritable.class);
	    MultipleOutputs.addNamedOutput(job, "testing", TextOutputFormat.class, Text.class, NullWritable.class);
	    MultipleOutputs.addNamedOutput(job, "recent23", TextOutputFormat.class, Text.class, NullWritable.class);
	    MultipleOutputs.addNamedOutput(job, "recent13", TextOutputFormat.class, Text.class, NullWritable.class);
	    job.waitForCompletion(true);

	 }
}
