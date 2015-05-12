package test;


import java.util.Arrays;
import java.util.Date;
import java.util.Random;
import java.io.File;

import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;

import weka.classifiers.trees.RandomForestSplitCriterionSelection;

public class PlotAccComparison {

    /**
     * @param args
     */
    public static void main(String[] args) {
        try{
        	long start, elapsed;
        	
        	// Instantiate a Date object
          Date date = new Date();
           
          // display time and date using toString()
          System.out.println(date.toString() + "\n");
          
          int rndSeed = 1;          

          int CV = 10;          
          String dirStr = "/home/simone/workspace/weka-3-6-6/tests/test8subset/";
          String numTrees = "100";         
           
          File dir = new File(dirStr);
          String[] children = dir.list();
          Arrays.sort(children);

          if (children == null) {
            System.out.println("Either dir does not exist or is not a directory");          	
          } else {
        	  double[][] vals = new double[3][children.length];
            double[][] cv = new double[3][CV];
            int[][] winlosses = new int[3][3];
            
            for (int f=0; f<children.length; f++) {
		          // Get filename of file or directory
		          String filename = children[f];
		          String source = dirStr + "/" + filename;
		          Instances data = DataSource.read(source);
		          data.setClassIndex(data.numAttributes()-1);

		          // Standard Random Forest
		          
              RandomForestSplitCriterionSelection rf = new RandomForestSplitCriterionSelection();
              String[] options = {"-C", "mi", "-I", numTrees, "-num-slots", "4"};//, "-print"};
              rf.setOptions(options);
              
              start = System.currentTimeMillis();
              rf.buildClassifier(data);
              System.err.println(System.currentTimeMillis() - start);
              
              Evaluation eval = new Evaluation(data);
              eval.crossValidateModel(rf, data, CV, new Random(rndSeed));
              vals[0][f] = eval.pctCorrect();  
              
              // Grassberger Random forest 
              
              rf = new RandomForestSplitCriterionSelection();
              String[] options_gmi = {"-C", "gmi", "-I", numTrees, "-num-slots", "4"};//, "-print"};
              rf.setOptions(options_gmi);
              
              start = System.currentTimeMillis();
              rf.buildClassifier(data);System.err.println(System.currentTimeMillis() - start);
              
              eval = new Evaluation(data);
              eval.crossValidateModel(rf, data, CV, new Random(rndSeed));
              vals[1][f] = eval.pctCorrect();  
              
              // AMI Random forest 
              
              rf = new RandomForestSplitCriterionSelection();
              String[] options_ami = {"-C", "ami", "-I", numTrees, "-num-slots", "4"};//, "-print"};
              rf.setOptions(options_ami);
              
              start = System.currentTimeMillis();
              rf.buildClassifier(data);
              System.err.println(System.currentTimeMillis() - start);
              
              eval = new Evaluation(data);
              eval.crossValidateModel(rf, data, CV, new Random(rndSeed));
              vals[2][f] = eval.pctCorrect();
              
              //print results
              System.out.print(filename.subSequence(3, filename.length()-5) + " & ");
              //classes
              System.out.print(data.numClasses() + " & ");
              //attributes
              System.out.print((data.numAttributes()-1) + " & ");
              //records
              System.out.print(data.numInstances() + " & ");
              
              double stdMetric = Utils.roundDouble(vals[0][f],2);
              System.out.print(stdMetric + " & ");
        			
              for (int i=1; i <= 2; i++){
                double metric = Utils.roundDouble(vals[i][f],2);
                if (metric > stdMetric){
                  System.out.print("\\textbf{" + metric + "} & ");
                  winlosses[0][i]++;
                }else{
                  System.out.print(metric+ " & ");
                }
              }
              System.out.println("\\\\");                
              
            }
              
            for (int i=0; i < 3; i++){
            	for (int j=0; j < 3; j++)
            		if (j !=i)
            			System.out.print(winlosses[i][j] + " ");
            		else
            			System.out.print("- ");
            	System.out.println();
            }
            
            System.out.println();
            
            //ties
            for (int i=0; i < 3; i++){
            	for (int j=0; j < 3; j++){
            		int ties = children.length - ( winlosses[i][j] + winlosses[j][i]);
            		if (j >= i)
            			System.out.print(ties + " ");
            		else
            			System.out.print("- ");
            	}
            	System.out.println();
            }              
            
            System.out.println();
            
            // for Wilcoxon test R
                        
            for (int i=0; i < 3; i++){
            	for (int j=0; j < vals[0].length; j++)
            		System.out.print(vals[i][j] + ", ");
            	System.out.println();
            }
           
          	// Instantiate a Date object
            date = new Date();
             
            // display time and date using toString()
            System.out.println(date.toString() + "\n");
              
          }
        }catch(Exception e) { e.printStackTrace(); }
    }
}
