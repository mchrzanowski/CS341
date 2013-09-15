package commons;

import java.io.Serializable;

/**
 * A simple container for a triplet of ints.
 * @author mc2711
 *
 */
public class Triplet implements Serializable {
	
	private static final long serialVersionUID = -3161264192283463853L;
	public int first;
	public int second;
	public int third;
	
	public Triplet(int first, int second, int third){
		this.first = first;
		this.second = second;
		this.third = third;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + second;
		result = prime * result + first;
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
		Triplet other = (Triplet) obj;
		if (second != other.second)
			return false;
		if (first != other.first)
			return false;
		return true;
	}

}