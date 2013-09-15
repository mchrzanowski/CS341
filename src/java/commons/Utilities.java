package commons;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

import com.google.gson.Gson;

public abstract class Utilities {
  
  public static Map<String, String[]> getItemToCategoryMapping(File itemMapping) throws FileNotFoundException{
    Map<String, String[]> mapping = new HashMap<>();
    
    try (Scanner sc = new Scanner(itemMapping)){
      while (sc.hasNextLine()){
        String[] chunks = sc.nextLine().split("\t");
        String item = chunks[0];
        String[] categories = chunks[1].split("#");
        
        mapping.put(item, categories);
      }
    }
    
    return mapping;
    
  }

  public static Map<Integer, String> getItemToNameyMapping(File itemMapping) throws FileNotFoundException{
    Map<Integer, String> mapping = new HashMap<>();    
    try (Scanner sc = new Scanner(itemMapping)){
      while (sc.hasNextLine()){
        String[] chunks = sc.nextLine().split("\t");
        StringBuilder builder = new StringBuilder();
        for (int i = 1; i < chunks.length; i++){
          builder.append(chunks[i] + " ");
        }
        mapping.put(Integer.parseInt(chunks[0]), builder.toString());
      }
    }
    
    return mapping;
    
  }

  public class ItemModel {
    public String item_id;
    public String title;
  }

  public static void createItemToNameyFile(String input, String output) throws Exception {
    Gson gson = new Gson();
    BufferedWriter writer = new BufferedWriter(new FileWriter(output));
    try (BufferedReader reader = new BufferedReader(new FileReader(input))) {
      String line = null;
      while ((line = reader.readLine()) != null) {
        ItemModel model = gson.fromJson(line, ItemModel.class);
        writer.write(String.format("%s\t%s\n", model.item_id, model.title));
      }
    }
    writer.close();
  }

  public static void main(String[] args) throws Exception {
    Map<Integer, String> lol = getItemToNameyMapping(new File("/Users/polak/input_data/item_to_names"));
    int i = 0;
    for (int l : lol.keySet()){
      System.out.println(l + " : " +  lol.get(l));
      i++;
      if (i > 1000) break;
      
    }
  }
}
