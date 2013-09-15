package commons;

import java.io.Serializable;

/**
 * A simple container for a triplet of ints.
 * @author mc2711
 *
 */
public class RawTriplet implements Serializable {
	
	private static final long serialVersionUID = -3161264192283463853L;
	public String first;
	public String second;
	public int third;
	
	public RawTriplet(String first, String second, int third){
		this.first = first;
		this.second = second;
		this.third = third;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((first == null) ? 0 : first.hashCode());
		result = prime * result + ((second == null) ? 0 : second.hashCode());
		result = prime * result + third;
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
		RawTriplet other = (RawTriplet) obj;
		if (first == null) {
			if (other.first != null)
				return false;
		} else if (!first.equals(other.first))
			return false;
		if (second == null) {
			if (other.second != null)
				return false;
		} else if (!second.equals(other.second))
			return false;
		if (third != other.third)
			return false;
		return true;
	}

}