package commons;

/**
 * A simple container class for a pair of ints.
 * @author mc2711
 *
 */
public class Pair {
	
	public int first;
	public int second;
	
	public Pair(int first, int second){
		this.first = first;
		this.second = second;
	}

	public int getSecond() {
		return second;
	}

	public int getFirst() {
		return first;
	}
	
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + first;
		result = prime * result + second;
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
		if (first != other.first)
			return false;
		if (second != other.second)
			return false;
		return true;
	}

}
