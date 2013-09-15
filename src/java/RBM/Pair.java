package RBM;

public class Pair implements Comparable<Pair> {

	public int item;
    public int rating;
    
    public Pair(int item, int rating) {
       this.item = item;
       this.rating = rating;
    }

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + item;
		result = prime * result + rating;
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Pair other = (Pair) obj;
		if (item != other.item)
			return false;
		if (rating != other.rating)
			return false;
		return true;
	}
	
	@Override
	public String toString() {
		return "Pair [item=" + item + ", rating=" + rating + "]";
	}

	@Override
	public int compareTo(Pair o) {
		if (this.item == o.item && this.rating == o.rating){
			return 0;
		}
		else if (this.item == o.item && this.rating < o.rating){
			return -1;
		}
		else {
			return 1;
		}
	}
      
}