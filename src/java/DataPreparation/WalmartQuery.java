import java.util.ArrayList;
import java.util.List;

import org.json.JSONArray;
import org.json.JSONObject;


public class WalmartQuery {
	private JSONObject queryJSONObject;
	public String visiorid = ""; 
	public String wmsessionid = ""; 
	public String searchtimestamp = "";
	ResultItemRating[] resultItemRatings;
	
	public WalmartQuery(JSONObject pQueryJSONObject) {
		queryJSONObject = pQueryJSONObject;
		visiorid = queryJSONObject.getString("visitorid");
		wmsessionid = queryJSONObject.getString("wmsessionid");
		searchtimestamp = queryJSONObject.getString("searchtimestamp");
		resultItemRatings = getQueryItemResultWithRating();
	}
	
	public ResultItemRating[] getQueryItemResultWithRating() {
		   JSONArray shownitems = queryJSONObject.getJSONArray("shownitems");
    	   JSONArray clicks = queryJSONObject.getJSONArray("clicks");
    	   
    	   List<ResultItemRating> resultItemRatingList = new ArrayList<ResultItemRating>();
    	   for(int i=0;i<shownitems.length();i++) {
    		  ResultItemRating resultItemRating = new ResultItemRating(shownitems.getLong(i));
    		  for(int j=0;j<clicks.length();j++) {
    			  if(resultItemRating.itemId == Long.parseLong(clicks.getJSONObject(j).getString("ItemId"))) {
    				  String ordered = clicks.getJSONObject(j).getString("Ordered");
    				  String incart = clicks.getJSONObject(j).getString("InCart");
    				  if(ordered.equals("true")) resultItemRating.finalRating = Label.ORDERED;
    				  else if(incart.equals("true")) resultItemRating.finalRating = Label.INCART;
    				  else resultItemRating.finalRating=Label.CLICKED;
    				  break;
    			  }
    		  }
    		  resultItemRatingList.add(resultItemRating);
    	   }
    	   return resultItemRatingList.toArray(new ResultItemRating[0]);
	}
	
	
	class ResultItemRating {
		long    itemId;
		Label   finalRating = Label.SHOWN;
		
		public ResultItemRating(long pItemId) {
			itemId = pItemId;
		}
	}
	
}
