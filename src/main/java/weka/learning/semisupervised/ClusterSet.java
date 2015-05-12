package weka.learning.semisupervised;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import weka.classifiers.Evaluation;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Clusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.gui.beans.xml.XMLBeans;

public class ClusterSet {

    private ArrayList<Integer> group_1 = new ArrayList<Integer>();
	private ArrayList<Integer> group_2 = new ArrayList<Integer>();	
	private Clusterer simpleCluster;
	private Instances set; 
	private int classIndex;
	
	public void SetClusterer(Clusterer c){
		simpleCluster = c;
	}
	
	private Instances loadData(String path){
		
		 // load data
		try {
			 BufferedReader reader = new BufferedReader(
                     new FileReader(path));
			 Instances data = new Instances(reader);
			 reader.close();
			 // setting class attribute
			// data.setClassIndex(data.numAttributes() - 1);;
				
			 
	    NumericToNominal filter = new NumericToNominal();
		filter.setInputFormat(data);
		 		 	   
	    return Filter.useFilter(data, filter);
		
		} catch (Exception e) {			
			
			e.printStackTrace();			
			return null;
			
		} 
	}
	
	private Instances getTestSet(int iLength, int jLength){
		
		ArrayList<Attribute> att = new ArrayList<Attribute>();
	
		for ( int i = 0; i<jLength ; i++ ) {
			att.add( new Attribute(new String("A"+i)));		
		}
				
		
		Instances set = new Instances("Test set",att,1);
		
		for(int i= 1; i<= iLength;i++){
			Instance iExample = new DenseInstance(jLength); 
			
			for(int j= 1; j<= jLength;j++){
								
				iExample.setValue(att.get(j-1),Double.parseDouble(j+"."+i));       							
			}	
			set.add(iExample);
		}
		
		System.out.println(set.toString());
		
		return set;
	}
	
	private Instances reverseInstances(Instances setOgirin){
	
		ArrayList<Attribute> att = new ArrayList<Attribute>();
		
		for ( int i = 0; i<setOgirin.numInstances(); i++ ) {
			att.add( new Attribute(new String("A"+i)));			
		}			
						
		Instances set = new Instances("Reverse set",att,setOgirin.numAttributes());
		
		for(int i= 0; i<setOgirin.numAttributes();i++){
			Instance iExample = new DenseInstance(setOgirin.numInstances()); 
			
			for(int j= 0; j<setOgirin.numInstances();j++){
				iExample.setValue(att.get(j),setOgirin.get(j).value(i));    
			}
			set.add(iExample);
		}
			
		return set;
	}
	
	
	public void buildClusterer(Instances ins, int classIndex){
		
		this.classIndex = classIndex;
		Instances setCluster = ins;
		set = ins;
		setCluster = reverseInstances(setCluster);
		
		if (simpleCluster == null) {
			simpleCluster = new SimpleKMeans();			
		}
		
		try {
			group_1.clear();
			group_2.clear();
			
			simpleCluster.buildClusterer(setCluster);

	        for(int i= 0 ;i <setCluster.numInstances(); i++){
	        	
	        	int v = simpleCluster.clusterInstance(setCluster.get(i));    
	        	
	        	if (v == 0 ){
	        		group_1.add(i);
	        	}
	        	if (v == 1 ){
	        		group_2.add(i);
	        	}
	        		        
	        	
	        }			
	        
        } catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	

    public ArrayList<Boolean> getListaAtributosUm(){    	
        
    	ArrayList<Boolean> listwhite = new ArrayList<Boolean>();    	                   
        for(int i = 0; i<set.numAttributes(); i++){
        	listwhite.add(false);
        }
    	
        for(Integer i : group_1){
        	listwhite.set(i, true);
        }
        
        listwhite.set(classIndex,true);
        
        return listwhite;
    }
    
    
   public ArrayList<Boolean> getListaAtributosDois(){    	
        
    	ArrayList<Boolean> listwhite = new ArrayList<Boolean>();    	                   
    	for(int i = 0; i<set.numAttributes(); i++){
        	listwhite.add(false);
        }
    	
        for(Integer i : group_2){
        	listwhite.set(i, true);
        }
        
        listwhite.set(classIndex,true);
        
        return listwhite;
    }
	

	
	public static void main (String args[]){
		
		ClusterSet c = new ClusterSet();
		
		Instances setCluster = c.loadData("/home/jhony/Downloads/Lymphoma.arff");
		setCluster.setClassIndex(setCluster.numAttributes()-1);
		
		CoTrainingClassifier cTraining = new CoTrainingClassifier();
		try {
			cTraining.buildClassifier(setCluster);
			Evaluation eval = new Evaluation(setCluster);
			
			eval.evaluateModel(cTraining, setCluster);
			
			System.out.println(eval.toSummaryString());
			
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
}
