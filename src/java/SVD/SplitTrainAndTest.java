package SVD;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

public class SplitTrainAndTest {
	public static void main(String[] args) {
		List list = new ArrayList();
		String input_training_file = "D:\\Stanford\\cs341\\svd_input";		
		HashMap<String, Integer> itemsDict = new HashMap<String, Integer>();
		HashMap<String, Integer> usersDict = new HashMap<String, Integer>();

		try {
			String s;
			FileReader fr = new FileReader(input_training_file);
			BufferedReader br = new BufferedReader(fr);

			Integer totalRecords = new Integer(0);
			while ((s = br.readLine()) != null) {
				list.add(s);
				totalRecords++;
			}
			br.close();
			fr.close();
			Collections.shuffle(list);
			
			String training_file = "D:\\Stanford\\cs341\\svd_training";
			FileWriter fw = new FileWriter(training_file);
			BufferedWriter bw = new BufferedWriter(fw);
            for(int i=0;i<list.size()*.8;i++) {
            	bw.write((String)list.get(i)+"\n");
            }
			bw.close();
			fw.close();
			
			String test_file = "D:\\Stanford\\cs341\\svd_test";
			fw = new FileWriter(test_file);
			bw = new BufferedWriter(fw);
            for(int i=(int)(list.size()*.8);i<list.size();i++) {
            	bw.write((String)list.get(i)+"\n");
            }
			bw.close();
			fw.close();
		}catch(Exception e){
			System.out.println("Exception ..");
		}
	}
}
