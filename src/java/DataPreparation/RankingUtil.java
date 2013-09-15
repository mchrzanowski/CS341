import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer.Context;

public class RankingUtil {

	public static void writeRecord(Context context, 
								   Text outputKey,
								   String token1, 
								   String token2, 
								   int rating, 
								   String searchtimestamp,
								   String wmsessionid) {
		    boolean isDuplicatesAllowed = Boolean.valueOf(context.getConfiguration().get("DUPLICATES_ALLOWED"));
			try {
				if (!isDuplicatesAllowed) {
					outputKey.set(token1 + "\t" + token2 + "\t" + rating + "\t" + searchtimestamp + "\t" + wmsessionid);
					context.write(outputKey, NullWritable.get());
			} else {
				if (rating == 2) {
					outputKey.set(token1 + "\t" + token2 + "\t" + 1 + "\t" + searchtimestamp + "\t" + wmsessionid);
					context.write(outputKey, NullWritable.get());
					outputKey.set(token1 + "\t" + token2 + "\t" + 2 + "\t" + searchtimestamp + "\t" + wmsessionid);
					context.write(outputKey, NullWritable.get());
				} else if (rating == 3) {
					outputKey.set(token1 + "\t" + token2 + "\t" + 1 + "\t" + searchtimestamp + "\t" + wmsessionid);
					context.write(outputKey, NullWritable.get());
					outputKey.set(token1 + "\t" + token2 + "\t" + 2 + "\t" + searchtimestamp + "\t" + wmsessionid);
					context.write(outputKey, NullWritable.get());
					outputKey.set(token1 + "\t" + token2 + "\t" + 3 + "\t" + searchtimestamp + "\t" + wmsessionid);
					context.write(outputKey, NullWritable.get());
				} else if (rating == 4) {
					outputKey.set(token1 + "\t" + token2 + "\t" + 1 + "\t" + searchtimestamp + "\t" + wmsessionid);
					context.write(outputKey, NullWritable.get());
					outputKey.set(token1 + "\t" + token2 + "\t" + 2 + "\t" + searchtimestamp + "\t" + wmsessionid);
					context.write(outputKey, NullWritable.get());
					outputKey.set(token1 + "\t" + token2 + "\t" + 3 + "\t" + searchtimestamp + "\t" + wmsessionid);
					context.write(outputKey, NullWritable.get());
					outputKey.set(token1 + "\t" + token2 + "\t" + 4 + "\t" + searchtimestamp + "\t" + wmsessionid);
					context.write(outputKey, NullWritable.get());
				} else {
					outputKey.set(token1 + "\t" + token2 + "\t" + 1 + "\t" + searchtimestamp + "\t" + wmsessionid);
					context.write(outputKey, NullWritable.get());
				}
			}
		} catch (Exception e) {
			System.out.println("Error writing rankign record.." + e.getMessage());
		}
	}
	
	public static PairRatingRecord getPairRating(Iterable<Text> values) {
		PairRatingRecord pairRatingRecord = new PairRatingRecord();
		
		for (Text line : values) {
			String[] valuetokens = line.toString().split("#");
			if(Integer.parseInt(valuetokens[0])>pairRatingRecord.rating.getId()) {
				pairRatingRecord.rating=Label.valueOf(Integer.parseInt(valuetokens[0])); 
				pairRatingRecord.wmsessionid=valuetokens[1];
				pairRatingRecord.searchtimestamp=valuetokens[2];
			}
			else if(Integer.parseInt(valuetokens[0]) == pairRatingRecord.rating.getId()) {
				pairRatingRecord.wmsessionid=valuetokens[1];
				pairRatingRecord.searchtimestamp=valuetokens[2];
			}
		}
		
		return pairRatingRecord;
	}	
}

class PairRatingRecord {
	Label rating = Label.SHOWN;;
	String wmsessionid;
	String searchtimestamp;
}
