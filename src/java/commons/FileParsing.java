package commons;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;
import java.util.Collections;

public abstract class FileParsing {

    /**
     * Read in the User-Item input file. Translate raw numbers to model-specific numbers.
     * Pass in two Map<String, Integer> data structures to capture the mappings
     * between the first two IDs to internal IDs for matrix operations.
     * Pass in a Set<Integer> to capture all labels seen in the input.
     * 
     * Pass in a boolean indicating whether this is training data or not.
     * If it is, then the method will actually populate the data structures you provided. Otherwise,
     * the method will only use the data structures to filter the testing data for data you've already
     * seen during training and will also make sure the ID mappings are preserved.
     * 
     * Assumes input is lines of form FIRST_ARG\tSECOND_ARG\tRATING
     * @param trainingFile
     * @param secondArgumentMapping
     * @param firstArgumentMapping
     * @param labels
     * @param trainingData
     * @return
     */
    public static List<Triplet> parseInputFile(File trainingFile, Map<String, Integer> firstArgumentMapping,
            Map<String, Integer> secondArgumentMapping, Set<Integer> labels, boolean trainingData, boolean shuffle){
        
        List<Triplet> inputData = new LinkedList<>();
        
        try (Scanner fileScanner = new Scanner(trainingFile)){
            while (fileScanner.hasNext()){
                String line = fileScanner.nextLine();
                
                String[] chunks = line.split("\t");
                if (trainingData && ! secondArgumentMapping.containsKey(chunks[1])){
                    secondArgumentMapping.put(chunks[1], secondArgumentMapping.size());
                }
                if (trainingData && ! firstArgumentMapping.containsKey(chunks[0])){
                    firstArgumentMapping.put(chunks[0], firstArgumentMapping.size());
                }
                
                int rating = Integer.parseInt(chunks[2]);
                
                if (trainingData && ! labels.contains(rating)){
                    labels.add(rating);
                }
                
                // only applicable for testing.
                int firstArgument = Constants.UNKNOWN;
                int secondArgument = Constants.UNKNOWN;
                if (firstArgumentMapping.containsKey(chunks[0])){
                    firstArgument = firstArgumentMapping.get(chunks[0]);
                }
                if (secondArgumentMapping.containsKey(chunks[1])){
                    secondArgument = secondArgumentMapping.get(chunks[1]);
                }                   
                inputData.add(new Triplet(firstArgument, secondArgument, rating));
                
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            System.exit(1);
        }
        
        if (shuffle){
            Collections.shuffle(inputData);
        }
        
        return inputData;
    }
    
    /**
     * Pretty much do the same thing as the above method, but also fill in
     * reverse lookup maps.
     * @param trainingFile
     * @param firstArgumentMapping
     * @param reverseFirstArgumentMapping
     * @param secondArgumentMapping
     * @param reverseSecondArgumentMapping
     * @param labels
     * @param trainingData
     * @param shuffle
     * @return
     */
    public static List<Triplet> parseInputFile(File trainingFile, Map<String, Integer> firstArgumentMapping,
            Map<Integer, String> reverseFirstArgumentMapping,
            Map<String, Integer> secondArgumentMapping, 
            Map<Integer, String> reverseSecondArgumentMapping,
            Set<Integer> labels, boolean trainingData, boolean shuffle){
        
        List<Triplet> inputData = new LinkedList<>();
        
        try (Scanner fileScanner = new Scanner(trainingFile)){
            while (fileScanner.hasNext()){
                String line = fileScanner.nextLine();
                
                String[] chunks = line.split("\t");
                
                if (trainingData && ! firstArgumentMapping.containsKey(chunks[0])){
                    firstArgumentMapping.put(chunks[0], firstArgumentMapping.size());
                    reverseFirstArgumentMapping.put(firstArgumentMapping.size() - 1, chunks[0]);
                }
                if (trainingData && ! secondArgumentMapping.containsKey(chunks[1])){
                    secondArgumentMapping.put(chunks[1], secondArgumentMapping.size());
                    reverseSecondArgumentMapping.put(secondArgumentMapping.size() - 1, chunks[1]);
                }
                
                int rating = Integer.parseInt(chunks[2]);
                
                if (trainingData && ! labels.contains(rating)){
                    labels.add(rating);
                }
                
                // only applicable for testing.
                int firstArgument = Constants.UNKNOWN;
                int secondArgument = Constants.UNKNOWN;
                if (firstArgumentMapping.containsKey(chunks[0])){
                    firstArgument = firstArgumentMapping.get(chunks[0]);
                }
                if (secondArgumentMapping.containsKey(chunks[1])){
                    secondArgument = secondArgumentMapping.get(chunks[1]);
                }                   
                inputData.add(new Triplet(firstArgument, secondArgument, rating));
                
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            System.exit(1);
        }
        
        if (shuffle){
            Collections.shuffle(inputData);
        }
        return inputData;
    }
    

}
