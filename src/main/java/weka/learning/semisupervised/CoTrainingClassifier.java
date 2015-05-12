package weka.learning.semisupervised;

import java.util.ArrayList;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;

public class CoTrainingClassifier extends AbstractClassifier  {

	 private Instances m_set;
	 private Instances subSetL1;
	 private Instances subSetL2; 
	 private Classifier H1;
	 private Classifier H2;
	 private ArrayList<Boolean> listaUm;
	 private ArrayList<Boolean> listaDois;
	 private int classIndex;
	 
	/**
	 * 
	 */
	private static final long serialVersionUID = 198763782883937L;

	public void buildClassifier(Instances data) throws Exception {
	     
		m_set = new Instances(data);
	}
	
	public void setSettings(Classifier h1, Classifier h2,ArrayList<Boolean> listaUm,ArrayList<Boolean> listaDois,int classIndex){
		
		H1 = h1;
		H2 = h2;
		this.listaUm = listaUm;
		this.listaDois = listaDois;
		this.classIndex = classIndex;
		
	}
	
	 
	 public double classifyInstance(Instance instance) throws Exception {
		 		 		 
		Instances set = new Instances(m_set, 1);
		set.setClassIndex(classIndex);
		set.clear();
		
		set.add(instance);
		
		//SubSetL1 reprensenta o set L'1 que contém uma parte dos atributos selecionados pelo usuário.
		subSetL1 = new Instances(set);
		subSetL1.setClassIndex(classIndex);
		
		for (int i = listaUm.size()-1; i >= 0  ; i--) {
			
			if (!listaUm.get(i)){
				subSetL1.deleteAttributeAt(i);
			}
		}

        //SubSetL2 reprensenta o set L'2 que contém uma parte dos atributos selecionados pelo usuário.
		subSetL2 = new Instances(set);	
		subSetL2.setClassIndex(classIndex);
		
		for (int i = listaDois.size()-1; i >= 0  ; i--) {
			
			if (!listaDois.get(i)){
				subSetL2.deleteAttributeAt(i);
			}
		}
						
		
        // CLASSIFICANDO A INSTÂNCIA QUE ESTÁ EM SUB SET L'   		
		double classe = 0;
		double grupoClasses[] ; 
		try {
			classe = H1.classifyInstance(subSetL1.instance(0));
			grupoClasses = H1.distributionForInstance(subSetL1.instance(0));
			
		} catch (Exception e) {
							
			e.printStackTrace();
			return -1;
		}
		
		
        String valor = subSetL1.instance(0).attribute(subSetL1.classIndex()).value((int) classe);
        subSetL1.instance(0).setValue(subSetL1.classIndex(),valor);
        // fim de classificação
        
        
        // CLASSIFICANDO A INSTÂNCIA QUE ESTÁ EM SUB SET L"		
      		double classe2 = 0;
      		double grupoClasses2[];
      		try {
      			classe2 = H2.classifyInstance(subSetL2.instance(0));
      			grupoClasses2 = H2.distributionForInstance(subSetL2.instance(0));
      		} catch (Exception e) {
      							
      			e.printStackTrace();
      			return -1;
      		}
      		
      		
        valor = subSetL2.instance(0).attribute(subSetL2.classIndex()).value((int) classe);
        subSetL2.instance(0).setValue(subSetL2.classIndex(),valor);
        // fim de classificação
		
		
		String u1Class= subSetL1.instance(0).stringValue(subSetL1.classAttribute());
		String u2Class= subSetL2.instance(0).stringValue(subSetL2.classAttribute());
		
		double somaClasses[]= new double[subSetL1.numClasses()];
		
		// Somando o grau de pertinência de todas as classes
		for ( int j= 0 ; j<subSetL1.numClasses();j++){
			
			 somaClasses[j] = grupoClasses[j]+grupoClasses2[j];
		}
		
		double max=0;
		int index_max =0;
		boolean empate = false;
		
		// Encontrando a maior soma entre os classificadores
		for ( int j= 0 ; j<subSetL1.numClasses();j++){
			 
			if (somaClasses[j] == max){
				empate= true;				
			}
			
			if (somaClasses[j]>max){
				max = somaClasses[j];
				index_max= j;
			}
			
			
		}					
		
	  // Caso ocorra um empate.	
	  if (empate){
			if ( u1Class.equals(u2Class)){
			    return classe;
			}else{
			     
				if (classe>classe2){
					return classe;
				}else{
					return classe2;
				}	
				
				
				
			}				 
		 }else{
			 			 
			 return  index_max;
			
			 
		 }
	 }

}

