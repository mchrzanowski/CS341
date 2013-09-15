package SVD;

public class ItemStats implements Comparable {
    long itemId;
    int totalOrderedCount;
    int totalClickedCount;
    int totalShowedCount;
    int totalCount;
    
    public ItemStats(long itemId) {
     this.itemId = itemId;	
    }
    
	public boolean equals(Object o) {
		if(this.itemId == ((ItemStats)o).itemId) return true;
		else return false;
	}

	public int hashCode() {
		return (int)itemId%100;
	}


	public int compareTo(Object o) {
       if((this.totalOrderedCount+this.totalClickedCount) > (((ItemStats)o).totalOrderedCount + ((ItemStats)o).totalClickedCount ))
		return -1;
       else if((this.totalOrderedCount+this.totalClickedCount) == (((ItemStats)o).totalOrderedCount + ((ItemStats)o).totalClickedCount ))
   		return 0;
       else
    	return 1;
	}
}

