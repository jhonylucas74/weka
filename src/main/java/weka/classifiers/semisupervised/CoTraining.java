package weka.classifiers.semisupervised;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.clusterers.Clusterer;
import weka.clusterers.EM;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.learning.semisupervised.ClusterSet;
import weka.learning.semisupervised.CoTrainingClassifier;
import weka.learning.semisupervised.QuickSort;

public class CoTraining extends AbstractClassifier implements OptionHandler, WeightedInstancesHandler, TechnicalInformationHandler{

	protected Classifier classifierBaseOne;
	protected Classifier classifierBaseTwo;
	
	protected int percentUnlabeledInstances ;
	protected int incorporatePerCycle;
	
	protected boolean separateAttributesWithCluster = false;
	protected Clusterer clustererBase;
	
	// Privates attributes
	private Instances setOriginal;
	private Instances setL;		
	private Instances setU;
	private Instances subSetL1;
	private Instances subSetL2;
	private Instances subSetU1; 		
	private Instances subSetU2;
	
	private int classIndex;
	
	private ArrayList<Boolean> listaAtributosUm = null;
	private ArrayList<Boolean> listaAtributosDois = null;
	
	private int set_porcent = 0; 
	
	public CoTraining(){
		classifierBaseOne = new IBk();
		classifierBaseTwo = new IBk();
		
		percentUnlabeledInstances = 20;
		incorporatePerCycle = 10;
		
		clustererBase =  new EM();
	}
	
	@Override
	public TechnicalInformation getTechnicalInformation() {
		// TODO Auto-generated method stub
		return null;
	}
	
	public double classifyInstance(Instance instance) throws Exception {
		
		CoTrainingClassifier cTraining = new CoTrainingClassifier();
		cTraining.buildClassifier(setOriginal);
		cTraining.setSettings(classifierBaseOne, classifierBaseTwo, listaAtributosUm, listaAtributosDois, classIndex);

		double classIndex = cTraining.classifyInstance(instance);        	
		String valor = instance.classAttribute().value((int) classIndex);
		instance.setValue(setL.classAttribute(), valor);
		
		return classIndex;
		  
	}
	
	@Override
	public void buildClassifier(Instances data) throws Exception {
		
		
		setOriginal = data;
		classIndex  = data.numAttributes()-1;
		
		if(separateAttributesWithCluster){
			ClusterSet c = new ClusterSet();
			c.SetClusterer(clustererBase);
			c.buildClusterer(data,classIndex);
			
			listaAtributosUm = c.getListaAtributosUm();
			listaAtributosDois = c.getListaAtributosDois();
		}else{
			randomAttributes();
		}
		

		// Set instances unlabeled in training set.   
		Instances trainingSetMissing = testeSelection(percentUnlabeledInstances);		
		
		/** Start the CoTraining */
		
		// Separate sets in L and U
		separaConjunto(trainingSetMissing);
		
		
		// While set U is not empty.
		while(setU.numInstances() > 0) {			
			// Separate the two views. L' and L'', U' and U''
			divideAtributos();
			// Build classifiers
			buildClassifiers();
			// generate U labeled with classifiers	
			classificaU(); 	    
			// Filter the best intances in U
			melhoresExemplos();	
		} 
		
		// the last training classifiers  	
		buildClassifiers();	
		    	
	}
	
	private void randomAttributes(){
		int length = setOriginal.numAttributes();
		
		listaAtributosUm = new ArrayList<Boolean>();
		listaAtributosDois = new ArrayList<Boolean>();  
		
        for(int i = 0; i< length; i++) listaAtributosUm.add(false);
        listaAtributosUm.set(length-1,true);
        
        for(int i = 0; i< length; i++) listaAtributosDois.add(false);
        listaAtributosDois.set(length-1,true);
        
        Random random = new Random();
        ArrayList<Integer> jaSaiu = new ArrayList<Integer>();
        jaSaiu.add(length-1);
        
        int t = (length-1) / 2;
        
        // For list One
        while(t > 0){
        	int index = random.nextInt(length);
        	
        	boolean unique = true;
        	
        	for(int j: jaSaiu){
        		if(j == index){ unique = false;}
        	}
        	
        	if(unique){
        	  jaSaiu.add(index);
        	  listaAtributosUm.set(index,true);
        	  t--;
        	}
        }
        
        t = (length-1) / 2;

        // For list Two
        while(t > 0){
        	int index = random.nextInt(length);
        	boolean unique = true;
        	
        	for(int j: jaSaiu){
        		if(j == index){ unique = false;}
        	}
        	
        	if(unique){
        	  jaSaiu.add(index);
        	  listaAtributosDois.set(index,true);
        	  t--;
        	}
        }
  
    
	}
	

	private Instances testeSelection(int porcentagem){
		
		// Copy the original Set.
		Instances trainingSetMissing = new Instances(setOriginal);
		
		// get integer of percent.
		int registros = (int) ((trainingSetMissing.numInstances()*porcentagem)/100);
		
		Random random = new Random();		
		int registro;
		
		ArrayList<Integer> saiu= new ArrayList<Integer>();
		boolean check;	
		
		while (registros>0 ){
			
			registro = random.nextInt(trainingSetMissing.numInstances());
			
			// Pick up ways somebody different.
			check = false;
			if (saiu.size()>0){				
				for(int i : saiu){
					if(i == registro){
						check= true;
						break;
					}
				}				
			}
		  
			if (!check){
				trainingSetMissing.get(registro).setMissing(classIndex);
				saiu.add(registro);
				registros--;
			}
		}
									
		return trainingSetMissing;
	}
	
	private void separaConjunto(Instances setFather) {
		
		// Create set L
		setL = new Instances(setFather);                 
        setL.setRelationName("Set L");
        setL.setClassIndex(classIndex);
        
        // Delete all instances without label.
        for( int i = 0; i < setL.numInstances(); i++ ) {
			if( setL.instance(i).isMissing(setL.classIndex())) {  
				setL.delete(i);
				i--;
			}	 
        }
        
        // Create set U                   
      	setU = new Instances(setFather);          
      	setU.setRelationName("Set U");   	
		setU.setClassIndex(classIndex);
      	
		 // Delete all instances with label.        
        for( int i = 0; i < setU.numInstances(); i++ ) {
	    	if( !setU.instance(i).isMissing(setU.classIndex()) ) {
	       		setU.delete(i);
	       		i--;  		 
	   	  	}
        }
        
	}
	
	private void divideAtributos() {
		// L'
		subSetL1 = new Instances(setL);
		subSetL1.setClassIndex(classIndex);
		// Delete attributes that the view don't have.
		for (int i = listaAtributosUm.size()-1; i >= 0  ; i--) {
			if (!listaAtributosUm.get(i)){
				subSetL1.deleteAttributeAt(i);
			}
		}
		// L''
		subSetL2 = new Instances(setL);
		// Delete attributes that the view don't have.
		for (int i = listaAtributosDois.size()-1; i >= 0  ; i--) {

			if (!listaAtributosDois.get(i)){
				subSetL2.deleteAttributeAt(i);
			}
		}
		// U'
		subSetU1 = new Instances(setU);
		subSetU1.setClassIndex(classIndex);
		// Delete attributes that the view don't have.
		for (int i = listaAtributosUm.size()-1; i >= 0  ; i--) {
			if (!listaAtributosUm.get(i)){
				subSetU1.deleteAttributeAt(i);
			}
		}
		// U''
		subSetU2 = new Instances(setU);
		// Delete attributes that the view don't have.
		for (int i = listaAtributosDois.size()-1; i >= 0  ; i--) {
			
			if (!listaAtributosDois.get(i)){
				subSetU2.deleteAttributeAt(i);
			}
		}
		 
	}
	
	private void buildClassifiers() {
	
		try {
			classifierBaseOne.buildClassifier(subSetL1); // utilizando o conjunto L para treinamento
			classifierBaseTwo.buildClassifier(subSetL2); // utilizando o conjunto L para treinamento
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}
	
	private void classificaU() {
		
		subSetU1.setRelationName("Sub set U'");              
        
        // Classify with classifier One.
        for(int i=0;i < subSetU1.numInstances();i++){
        	double classe = 0;
			try {
				classe = classifierBaseOne.classifyInstance(subSetU1.instance(i));
			} catch (Exception e) {			
				e.printStackTrace();
				return;
			}
	        String valor = subSetU1.instance(i).attribute(subSetU1.classIndex()).value((int) classe);
	        subSetU1.instance(i).setValue(subSetU1.classIndex(),valor);
        }		
                

		subSetU2.setRelationName("Sub set U'' ");              
        
		// Classify with classifier Two.
        for(int i=0;i < subSetU2.numInstances();i++){
        	double classe = 0;
			try {
				classe = classifierBaseTwo.classifyInstance(subSetU2.instance(i));
			} catch (Exception e) {
								
				e.printStackTrace();
				return;
			}
	        String valor = subSetU2.instance(i).attribute(subSetU2.classIndex()).value((int) classe);
	        subSetU2.instance(i).setValue(subSetU2.classIndex(),valor);
        }		
        
	}
	
	
	private void melhoresExemplos(){
		
		try {
		
		// Length of vector.
		int length = subSetU1.numInstances();
			
		// Vectors to store the index and the degree of relevance of intancias.
		double subSetU1Pertinencia[][]= new double[2][length];
		double subSetU2Pertinencia[][]= new double[2][length];
		// Vector to temporarily save the relevance of instances.
		double distClass[];   
		
		/*
		 * For set U'
		 */
		
		for(int i = 0; i < length; i++) {
			subSetU1Pertinencia[0][i] = i;
			distClass = classifierBaseOne.distributionForInstance(subSetU1.instance(i));
			
			// Select the best relevance value.
			double max = 0;
			for(int j = 0 ; j<subSetU1.numClasses();j++){				
				  if (distClass[j] > max){
					  max = distClass[j] ;
				  }
			}
			
		    subSetU1Pertinencia[1][i] = max;
		}
				
			
		/*
		 * For set U"
		 */
		
		for(int i = 0; i < length; i++) {
			subSetU2Pertinencia[0][i] = i;
			distClass = classifierBaseTwo.distributionForInstance(subSetU2.instance(i));
			
			// Select the best relevance value.
			double max = 0;
			
			for(int j = 0 ; j<subSetU2.numClasses();j++){
				  if (distClass[j] > max){
					  max = distClass[j] ;
				  }
			}
		    subSetU2Pertinencia[1][i] = max;
		}
		
		// Adding the maximum relevance value of the vectors thus creating a new vector.
		double subSetUPertinencia[][] = new double[2][length];
		
		for(int i = 0; i< length; i++){
			subSetUPertinencia[0][i] = i ;
		    
			String u1Class= subSetU1.instance(i).stringValue(subSetU1.classAttribute());
			String u2Class= subSetU1.instance(i).stringValue(subSetU1.classAttribute());
			
			if ( u1Class.equals(u2Class)){
				subSetUPertinencia[1][i] = subSetU1Pertinencia[1][i]+subSetU2Pertinencia[1][i];
			}else{
				subSetUPertinencia[1][i] = 0;
			}
		}
						
		// Order subsetUPertinencia 
		QuickSort sorter = new QuickSort();
		sorter.sort(subSetUPertinencia, length);
		
	 	/*
	 	 *  CLASSIFICANDO SET U
	 	 */
	
		if (set_porcent == 0){
		  set_porcent = (length*incorporatePerCycle)/100 ;				
		} 
		
		// Redefining percentage, final cycle of the algorithm.
		if (setU.numInstances()<set_porcent){
			set_porcent = setU.numInstances();
		}
					
		
		for( int i = length-1; i>(length-1)-set_porcent; i--){
			setU.instance((int)subSetUPertinencia[0][i]).setValue(classIndex,subSetU1.instance((int)subSetUPertinencia[0][i]).value(subSetU1.classIndex()) );
		}
								
		/*
		 *  REMOVENDO DE U E ADICIONANDO EM L
		 */
		
		for ( int  i = length-1; i>-1; i--){			
			if( !setU.instance(i).isMissing(setU.classAttribute().index())){ 
				 setL.add(setU.instance(i));
				 setU.delete(i);
			}
		}	   		
		
		}catch(Exception e){
			  e.printStackTrace();
		  }
	}
	
	// OPTIONS
	// --------------------------------------------------------------------------------------------
	
	/**
	* Parses a given list of options. <p/>
	*
	<!-- options-start -->
	* Valid options are: <p/>
	* 
	* <pre> -U
	*  Percent value of unlabeled instances. 
	*  (Default = 10)</pre>
	* 
	* <pre> -S
	*  For attribute selection with clustering. </pre>
	* 
	* <pre> -I
	*  Percent value to incorporate per cycle. </pre>
	* 
	* <pre> -H1
	*  Select the classifier one algorithm to use (default: weka.classifiers.lazy.IBk).
	* </pre>
	* 
	* <pre> -H2
	*  Select the classifier two algorithm to use (default: weka.classifiers.lazy.IBk).
	* </pre>
	*
	* <pre> -C
	*  The cluster base algorithm to use (default: weka.clusterers.EM).
	* </pre>
	* 
	<!-- options-end -->
	*/
	public Enumeration listOptions() {
		
		Vector newVector = new Vector();
	
		newVector.addElement(new Option(
				"\tSelect the classifier one algorithm to use (default: weka.classifiers.lazy.IBk).",
				"H1", 0,"-H1"));
		
		newVector.addElement(new Option(
				"\tSelect the classifier two algorithm to use (default: weka.classifiers.lazy.IBk).",
				"H2", 0,"-H2"));
				
		newVector.addElement(new Option(
				"\tFor attributes selection with clustering.",
				"S", 0,"-S"));		
		
		newVector.addElement(new Option(
				"\tPercent value to incorporate per cycle.",
				"I", 0,"-I"));
		
		newVector.addElement(new Option(
				"\tPercent value of unlabeled instances.",
				"U", 0,"-U"));
		
		newVector.addElement(new Option(
				"\tThe cluster base algorithm to use (default: weka.clusterers.EM).",
				"C", 0,"-C"));
		
				
		return newVector.elements();
	}
	
	public void setOptions(String[] options) throws Exception {
		
		setSeparateAttributesWithCluster(Utils.getFlag('S', options));
		
		String incorporateString = Utils.getOption('I', options);
		if (incorporateString.length() != 0) {
			setIncorporatePerCycle(Integer.parseInt(incorporateString));
		} else {
			setIncorporatePerCycle(10);
		}
		
		String percentUnlabeledInstances = Utils.getOption('U', options);
		if (percentUnlabeledInstances.length() != 0) {
			setPercentUnlabeledInstances(Integer.parseInt(percentUnlabeledInstances));
		} else {
			setPercentUnlabeledInstances(10);
		}
			
		// Classifier 1
		String classifierBaseOne = Utils.getOption('C', options);
		if (classifierBaseOne.length() != 0) {
			String classifierBaseS[] = Utils.splitOptions(classifierBaseOne);
			if(classifierBaseS.length == 0) { 
			throw new Exception("Invalid classifier algorithm " +
								"specification string."); 
	    }
		  String className = classifierBaseS[0];
		  classifierBaseS[0] = "";

		  setClassifierBaseOne( (Classifier)
					  Utils.forName( Classifier.class, 
									 className, 
									 classifierBaseS)
									);
		} else {
			setClassifierBaseOne(new IBk());
		}
		
		// Classifier 2
		String classifierBaseTwo = Utils.getOption('C', options);
		if (classifierBaseTwo.length() != 0) {
			String classifierBaseS[] = Utils.splitOptions(classifierBaseTwo);
			if(classifierBaseS.length == 0) { 
			throw new Exception("Invalid classifier algorithm " +
								"specification string."); 
	    }
		  String className = classifierBaseS[0];
		  classifierBaseS[0] = "";

		  setClassifierBaseTwo( (Classifier)
					  Utils.forName( Classifier.class, 
									 className, 
									 classifierBaseS)
									);
		} else {
			setClassifierBaseTwo(new IBk());
		}
		
		// Cluster Base
		String clustererBase = Utils.getOption('G', options);
		if (clustererBase.length() != 0) {
			String clustererBaseS[] = Utils.splitOptions(clustererBase);
			if(clustererBaseS.length == 0) { 
			throw new Exception("Invalid classifier algorithm " +
								"specification string."); 
			}
			String className = clustererBaseS[0];
			clustererBaseS[0] = "";
	
			setClustererBase( (Clusterer)
					  Utils.forName( Clusterer.class, 
									 className, 
									 clustererBaseS)
									);
		} else {
			setClustererBase(new EM());
		}
		
		Utils.checkForRemainingOptions(options);
	}
	
	public String [] getOptions() {
		
		String [] options = new String [11];
		int current = 0;
		
		if (separateAttributesWithCluster) {
		  options[current++] = "-S";
		}

		options[current++] = "-I"; options[current++] = "" + getIncorporatePerCycle();
		options[current++] = "-U"; options[current++] = "" + getPercentUnlabeledInstances();
		options[current++] = classifierBaseOne.getClass().getName();
		options[current++] = classifierBaseTwo.getClass().getName();
		options[current++] = clustererBase.getClass().getName();
		
		while (current < options.length) {
		  options[current++] = "";
		}
		
		return options;
	}
	
	// GETTERS AND SETTERS
	//---------------------------------------------------------------------------------------------
	
	public Classifier getClassifierBaseOne() {
		return classifierBaseOne;
	}

	public void setClassifierBaseOne(Classifier classifierBaseOne) {
		this.classifierBaseOne = classifierBaseOne;
	}

	public Classifier getClassifierBaseTwo() {
		return classifierBaseTwo;
	}

	public void setClassifierBaseTwo(Classifier classifierBaseTwo) {
		this.classifierBaseTwo = classifierBaseTwo;
	}

	public int getPercentUnlabeledInstances() {
		return percentUnlabeledInstances;
	}

	public void setPercentUnlabeledInstances(int percentUnlabeledInstances) {
		this.percentUnlabeledInstances = percentUnlabeledInstances;
	}

	public int getIncorporatePerCycle() {
		return incorporatePerCycle;
	}

	public void setIncorporatePerCycle(int incorporatePerCycle) {
		this.incorporatePerCycle = incorporatePerCycle;
	}

	public boolean isSeparateAttributesWithCluster() {
		return separateAttributesWithCluster;
	}

	public void setSeparateAttributesWithCluster(
			boolean separateAttributesWithCluster) {
		this.separateAttributesWithCluster = separateAttributesWithCluster;
	}

	public Clusterer getClustererBase() {
		return clustererBase;
	}

	public void setClustererBase(Clusterer clustererBase) {
		this.clustererBase = clustererBase;
	}
	
	

}
