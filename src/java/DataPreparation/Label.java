public enum Label {
	SHOWN(1), CLICKED(2), INCART(3), ORDERED(4);
	private int id;

	private Label(int pId) {
		id = pId;
	}

	public int getId() {
		return id;
	}
	
	public static Label valueOf(int pId) {
		if(pId==1) {
			return Label.SHOWN;
		}
		else if(pId==2) {
			return Label.CLICKED;
		}
		else if(pId==3) {
			return Label.INCART;
		}
		else if(pId==4) {
			return Label.ORDERED;
		}
		else
			return Label.SHOWN;
	}
}
