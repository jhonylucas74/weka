package test;

import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomForestSplitCriterionSelection;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class testRandomForestsSplitCriterionSelection {
	public static void main(String[] args){
		try{
			long start, elapsedTime;
			//String source = "/home/simone/workspace/weka-3-6-6/tests/test8/_p1mnist.arff";
			String source = "/home/simone/workspace/weka-3-6-6/files/UCI/glass.arff";
			Instances data = DataSource.read(source);
			data.setClassIndex(data.numAttributes()-1);
			
			String numTrees = "100";
				
			// Random Forest standard implementation
			RandomForest rf_std = new RandomForest();
			String[] options_std = {"-I", numTrees, "-num-slots", "4"};//, "-print"};
			rf_std.setOptions(options_std);
			start = System.nanoTime();   
			rf_std.buildClassifier(data);
			elapsedTime = System.nanoTime() - start;
			System.out.println("Elapsed time RF: " + elapsedTime/1E9 + " s\n");
			
			System.out.println(rf_std);
			
			Evaluation eval_std = new Evaluation(data);
			eval_std.crossValidateModel(rf_std, data, 10, new Random(1));
			System.out.println("Acc = " + eval_std.pctCorrect() + "\n");
			
			
			// Split selection Random Forest implementation
			RandomForestSplitCriterionSelection rf = new RandomForestSplitCriterionSelection();
			String[] options = {"-C", "ami", "-I", numTrees, "-num-slots", "4"};//, "-print"};
			rf.setOptions(options);
			start = System.nanoTime();   
			rf.buildClassifier(data);
			elapsedTime = System.nanoTime() - start;
			System.out.println("Elapsed time RF: " + elapsedTime/1E9 + " s\n");
			
			System.out.println(rf);
			
			Evaluation eval = new Evaluation(data);
			eval.crossValidateModel(rf, data, 10, new Random(1));
			System.out.println("Acc = " + eval.pctCorrect() + "\n");
			
		}catch(Exception e){e.printStackTrace();}
	}
}
