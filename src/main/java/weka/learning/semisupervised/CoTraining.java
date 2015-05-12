package weka.learning.semisupervised;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Random;

import javax.swing.JTextPane;

import weka.associations.AssociationRule;
import weka.associations.AssociationRules;
import weka.associations.Associator;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializedObject;
import weka.gui.GenericObjectEditor;
import weka.gui.Logger;
 
public class CoTraining extends Thread implements Runnable{
  
	 private volatile boolean isRunning = true;
	
	/** Atributos que devem ser passados obrigatoriamente para esta classe funcionar corretamente. */
	private Instances setOriginal = null ;
	private Classifier algoClassificacao1 = null;
	private Classifier algoClassificacao2 = null;
	private int classIndex ;
	private JTextPane m_OutText = null;
	private ArrayList<Boolean> listaAtributosUm = null;
	private ArrayList<Boolean> listaAtributosDois = null;
	
	
	private Instances setL; 			//Sub-conjunto de dados rotulados	
	private Instances setU; 			//Sub-conjunto de dados n�o-rotulados
	private Classifier classifierH1; 	//Classificador treinado a partir do cojunto L
	private Classifier classifierH2; 	//Classificador treinado a partir do cojunto L
	private Instances subSetL1; 		
	private Instances subSetL2; 		
	private Instances subSetU1; 		
	private Instances subSetU2; 			
	
	private Evaluation eTest  ;
	private Evaluation eTest2 ;
	
	/** Modo do teste dos classificadores 
	 *  0 training set
	 *  1 cross -validation
	 * */
	private int modTraining = 0;
	
	/** Atributos que seram utilizados para o modo cross - validation */
	private int seed = 30;
	private int folds = 10;
	
	/** Atributo para a remoção dos rótulos no inicio do treinamento */
	private int percentageUnlabeled = 23;
	
	private Logger m_Log; 
	
	/** Atributo booleano que indicará se será impresso o resultado do teste final da comparação
	 *  do conjunto gerado pelos classificadores */
	private boolean printFinalTeste = true;
	
	/** Atributo para a porcentagem do icremento por ciclo dos melhores exemplos*/
	private int percentMe = 10;
	
	/** Utilizada no método >> melhoresExemplos */
	private int set_porcent = 0; 	   // Porcentagem real das intâncias utilizada para selecionar os melhores exemplos.
	
	/*
	 * Atributos pra as regras de associação
	 */	
	private Associator associator;
	
	public CoTraining(Instances setOriginal,Classifier algoClassificacao1, Classifier algoClassificacao2,
			int classIndex,JTextPane m_OutText){
		
		this.setOriginal = setOriginal;
		this.algoClassificacao1 = algoClassificacao1;
		this.algoClassificacao2 = algoClassificacao2;
		this.classIndex= classIndex;
		this.m_OutText = m_OutText;
	}
	
	/** Caso seja escolhido esse construtor é necessário setar os parâmetros necessários para o funcionameto desse algoritmo.*/
	public CoTraining(){

	}
	
	public void kill() {
		isRunning = false;
	}
		
	
	public void run (){
				    	
		//Validar estradas caso esteja tudo bem o processo ir� continuar. 
		if (!validarEntradas()){
			return;
		}		
		
		if (!isRunning){ return; }
		
		m_Log.logMessage("Start training  Co-Training");
		m_Log.statusMessage("Start training  Co-Training");
		// Modificando o cojunto que será utilizado para treinamento , definindo uma porcentagem de registros com classe faltosa.   
		Instances trainingSetMissing = testeSelection(percentageUnlabeled);		
		
    	/** Iniciando Algoritmo do Self-Training */
    	
    	// Separa o conjunto de teste em L (rotulados) e U (não-rotulados)
		if (!isRunning){ return; }
		
    	separaConjunto(trainingSetMissing);
    	
    	if (!isRunning){ return; }    	
    	divideAtributos();    
    	
    	// Imprimindo da tela do us�ario
    	out(
    	"=== Run information === \n"+    	
    	"Relation:     "+setOriginal.relationName()+"\n"+
    	"Instances:    "+setOriginal.numInstances()+"\n"+
    	"With label:   "+setL.numInstances()+"\n"+
    	"No label:     "+setU.numInstances()+"\n"+
    	"Attributes:   "+setOriginal.numAttributes()+"\n\n"+
    	"Group 1"
    	);
    	
    	for(int i=0;i<subSetL1.numAttributes();i++ ){
    		
    		m_OutText.setText(m_OutText.getText()+"              "+subSetL1.attribute(i).name()+"\n");
    	}    
    	
    	out("\n\nGroup 2");
    	for(int i=0;i<subSetL2.numAttributes();i++ ){
    		
    		m_OutText.setText(m_OutText.getText()+"              "+subSetL2.attribute(i).name()+"\n");
    	}    
    	
   
    	/** Condição enquanto ainda houver elementos no conjunto de U */
    	while(setU.numInstances() > 0) {		
    		if (!isRunning){ return; }    		
    		// Dividindo os atributos do conjunto L e U, criando outros 4 novos subconjuntos. 
        	divideAtributos();        
	    	criaClassificadorH(algoClassificacao1, algoClassificacao2 );		
	    	classificaU(); 	// gerando U' a partir do classificador H	    		    	
	    	melhoresExemplos();	// selecionar as melhores int�ncias de U' e as adiciona em L
    	} /** Fim do while..*/
    	
    	// Treinando o classificador pela ultima vez.
    	if (!isRunning){ return; }    	
    	criaClassificadorH( algoClassificacao1, algoClassificacao2 );		
    	    	
     	m_Log.logMessage("End training classifiers");
    	m_Log.statusMessage("End training classifiers");
    	
    	// Dividindo o cojunto setOriginal para subSetL1 e subSetL2. Isso é necessário para que o teste possa ser realizado
        // com o cojunto original que contém todos os atributos com suas devidas classes conhecidas. 
    	if (!isRunning){ return; }    	
    	dividirAtributosL();       
    	
        /*
         *  Testando conjunto de dados utilizando cross - validation        
         */
        Evaluation evalH1=  null;
        Evaluation evalH2=  null;
        
        if (!isRunning){ return; }        
        
        if (modTraining == 1){
		        try {
		        	 evalH1 =  crossValidationTeste(subSetL1, algoClassificacao1);
		        	 evalH2 =  crossValidationTeste(subSetL2, algoClassificacao2);
				} catch (Exception e) {		
					e.printStackTrace();
				}
        }else{
        	evalH1 = eTest;
        	evalH2 = eTest2;        	
        }
        
    	// Saída destinada ao m_outText
    	m_OutText.setText(m_OutText.getText()+"\n"+"=== Summary ==="+"\n \n"+
    	"Scheme:       "+algoClassificacao1.getClass().toString()+"\n");
    	
    	// Resultado do test
    	imprimeResultadoH(evalH1);
    	m_OutText.setText(m_OutText.getText()+"\n Scheme:       "+algoClassificacao2.getClass().toString()+"\n");
    	imprimeResultadoH(evalH2);
    	    	
    	
		try {
			if (!isRunning){ return; }			
			CoTrainingClassifier cTraining = new CoTrainingClassifier();
						 			
			cTraining.buildClassifier(setOriginal);
			cTraining.setSettings(classifierH1, classifierH2, listaAtributosUm, listaAtributosDois, classIndex);
			setOriginal.setClassIndex(classIndex);
			
			Evaluation eval = crossValidationTeste(setOriginal,cTraining);		
			
			String strSummary = eval.toSummaryString("=== "+" Use two Classifiers  ===",
					false)+"\n"+eval.toMatrixString();
			
			out("\n"+strSummary+"\n");
			
		} catch (Exception e) { 
			e.printStackTrace();
		}
		    	    
    	
    	m_Log.logMessage("Start test final ");
    	m_Log.statusMessage("Start test final");
    	// Teste para verificar a qualidade de predição.    
    	if (printFinalTeste){
    		testeResult(trainingSetMissing);
    	}
    	
    	m_Log.logMessage("End of the execution of Algorithm Co-Training");
    	m_Log.statusMessage("OK!");
    	    	
    }
	
	/**  Separar um conjunto de instancias em dois sub-conjuntos:
	 *   Sub-conjunto L: instancias rótuladas
	 *   Sub-conjunto U: instancias não rótuladas
	 *  
	 *  @param setOriginal Instances - conjunto de dados original
	 */
	private void separaConjunto(Instances setFather) {
		
        
		//Cria o sub-conjunto L
		setL = new Instances(setFather); //cria uma cópia do conjunto original                 
        setL.setRelationName("Set L");     //nomeia o sub-conjunto L
        setL.setClassIndex(classIndex);
        
        //Percorre todo o sub-conjunto L para eliminar todos os elementos sem-r�tulos
        for( int i = 0; i < setL.numInstances(); i++ ) {
			if( setL.instance(i).isMissing(setL.classIndex())){ //verifica se o exemplo possui valor de classe 
				setL.delete(i);
				i--;
			}	 
        }
        
        //Cria o sub-conjunto U                   
      	setU = new Instances(setFather); //cria uma c�pia do conjunto original          
      	setU.setRelationName("Set U"); 	   //nomeia o sub-conjunto U    	
		setU.setClassIndex(classIndex);
      	
        //Percorre todo o sub-conjunto U para eliminar todos os elementos com r�tulos         
        for( int i = 0; i < setU.numInstances(); i++ ) {
        	//verifica se o valor do atributo classe � diferente de null
	    	if( !setU.instance(i).isMissing(setU.classIndex()) ) {
	       		setU.delete(i);
	       		i--;  		 
	   	  	}
        }
        

        
	}
	
	/**  Gera o classificador H com base no conjunto dos dados rotulados
	 *  
	 *   @param  algClassificacao Classifier - o algoritmo de classificação definido pelo usuário   
	 *   @return Evaluation - avaliação feita do classificador
	 */
	private void criaClassificadorH(Classifier algClassificacao1, Classifier algClassificacao2) {
								
		classifierH1 = algClassificacao1;
		classifierH2 = algClassificacao2;
		
		eTest = null;
		try {
			classifierH1.buildClassifier(subSetL1); // utilizando o conjunto L para treinamento
	        eTest = new Evaluation(subSetL1);
	        eTest.evaluateModel(classifierH1, subSetL1); // testando o modelo gerado a partir do treinamento
		} catch (Exception e) {
			
			m_Log.logMessage(e.getMessage());
	    	m_Log.statusMessage("Classifier Error in H1 - Check the compatibility of the classifier");
	    	
			e.printStackTrace();
			
			interrupt(); 
		}
		
		
		eTest2 = null;
		try {
			classifierH2.buildClassifier(subSetL2); // utilizando o conjunto L para treinamento
	        eTest2 = new Evaluation(subSetL2);
	        eTest2.evaluateModel(classifierH2, subSetL2); // testando o modelo gerado a partir do treinamento
		} catch (Exception e) {
			
			m_Log.logMessage(e.getMessage());
	    	m_Log.statusMessage("Classifier Error in H2 - Check the compatibility of the classifier");
	    		
	    	interrupt(); 
	    	
			e.printStackTrace();
						
		}
		
	}
	
	/**
	 * 
	 *  
	 * @return void
	 */
	private void divideAtributos() {
		
		//SubSetL1 reprensenta o set L'1 que contém uma parte dos atributos selecionados pelo usuário.
		subSetL1 = new Instances(setL);
		subSetL1.setClassIndex(classIndex);
		
		for (int i = listaAtributosUm.size()-1; i >= 0  ; i--) {
			
			if (!listaAtributosUm.get(i)){
				subSetL1.deleteAttributeAt(i);
			}
		}
		
		//SubSetL2 reprensenta o set L'2 que contém uma parte dos atributos selecionados pelo usuário.
				subSetL2 = new Instances(setL);
				
				for (int i = listaAtributosDois.size()-1; i >= 0  ; i--) {
					
					if (!listaAtributosDois.get(i)){
						subSetL2.deleteAttributeAt(i);
					}
				}
		 
		//SubSetU1 reprensenta o set U'1 que cont�m uma parte dos atributos selecionados pelo usuário.
		subSetU1 = new Instances(setU);
		subSetU1.setClassIndex(classIndex);
		
		for (int i = listaAtributosUm.size()-1; i >= 0  ; i--) {
			
			if (!listaAtributosUm.get(i)){
				subSetU1.deleteAttributeAt(i);
			}
		}
		
		//SubSetU2 reprensenta o set U'2 que cont�m uma parte dos atributos selecionados pelo usuário.
				subSetU2 = new Instances(setU);
				
				for (int i = listaAtributosDois.size()-1; i >= 0  ; i--) {
					
					if (!listaAtributosDois.get(i)){
						subSetU2.deleteAttributeAt(i);
					}
				}
		 
	}
	
	/**
	 * 
	 */
	private void dividirAtributosL(){
		
		
		//SubSetL1 reprensenta o set L'1 que contém uma parte dos atributos selecionados pelo usuário.
				subSetL1 = new Instances(setOriginal);
				subSetL1.setClassIndex(classIndex);
				
				for (int i = listaAtributosUm.size()-1; i >= 0  ; i--) {
					
					if (!listaAtributosUm.get(i)){
						subSetL1.deleteAttributeAt(i);
					}
				}
		
		//SubSetL2 reprensenta o set L'2 que contém uma parte dos atributos selecionados pelo usuário.
				subSetL2 = new Instances(setOriginal);
				subSetL2.setClassIndex(classIndex);
				
				for (int i = listaAtributosDois.size()-1; i >= 0  ; i--) {
					
					if (!listaAtributosDois.get(i)){
						subSetL2.deleteAttributeAt(i);
					}
				}

	}
	
	
	/**
	 * 
	 */
	private void classificaU() {
			
        //Renomeando o sub conjunto U'
		subSetU1.setRelationName("Sub set U'");              
        
        //Classificando com H1
        for(int i=0;i < subSetU1.numInstances();i++){
        	double classe = 0;
			try {
				classe = classifierH1.classifyInstance(subSetU1.instance(i));
			} catch (Exception e) {
								
				e.printStackTrace();
				return;
			}
	        String valor = subSetU1.instance(i).attribute(subSetU1.classIndex()).value((int) classe);
	        subSetU1.instance(i).setValue(subSetU1.classIndex(),valor);
        }		
                
        //Renomeando o sub conjunto U'
		subSetU2.setRelationName("Sub set U'' ");              
        
        //Classificando com H2
        for(int i=0;i < subSetU2.numInstances();i++){
        	double classe = 0;
			try {
				classe = classifierH2.classifyInstance(subSetU2.instance(i));
			} catch (Exception e) {
								
				e.printStackTrace();
				return;
			}
	        String valor = subSetU2.instance(i).attribute(subSetU2.classIndex()).value((int) classe);
	        subSetU2.instance(i).setValue(subSetU2.classIndex(),valor);
        }		
        
	}
	
	/**
	 * 
	 */
	private void melhoresExemplos(){
		
		try {
		
		// Tamanho do vetor
		int length = subSetU1.numInstances();
			
		// Vetores para guardarem o index e o grau de pertinência das intancias.
		double subSetU1Pertinencia[][]= new double[2][length];
		double subSetU2Pertinencia[][]= new double[2][length];
		
		double distClass[]; // Vetor para guardar temporáriamente a pertinência da instância.  
		
		/*
		 * PARA SUB SET U'
		 */
		
		for(int i = 0; i < length; i++) {
			// Guardando na primeira coluna do vetor subSerU1Pertinencia o index do subSetU1.
			subSetU1Pertinencia[0][i] = i;
			// Guardando na segunda coluna do vetor o valor de maior pertinencia da instância 
			distClass = classifierH1.distributionForInstance(subSetU1.instance(i));
			
			// Selecionando o maior valor da pertinência
			double max = 0;
			for(int j = 0 ; j<subSetU1.numClasses();j++){				
				  if (distClass[j] > max){
					  max = distClass[j] ;
				  }
			}
		    // adicionando o maior valor de pertinência ao vetor.
		    subSetU1Pertinencia[1][i] = max;
		}
				
			
		/*
		 * PARA SUB SET U"
		 */
		
		for(int i = 0; i < length; i++) {
			// Guardando na primeira coluna do vetor subSerU2Pertinencia o index do subSetU1.
			subSetU2Pertinencia[0][i] = i;
			// Guardando na segunda coluna do vetor o valor de maior pertinencia da instância
			distClass = classifierH2.distributionForInstance(subSetU2.instance(i));
			
			// Selecionando o maior valor da pertinência
			double max = 0;
			
			for(int j = 0 ; j<subSetU2.numClasses();j++){
				  if (distClass[j] > max){
					  max = distClass[j] ;
				  }
			}
		    // adicionando o maior valor de pertinência ao vetor.
		    subSetU2Pertinencia[1][i] = max;
		}
		
		// Somando o máximo valor de pertinencia dos vetores criando assim um novo vetor
		double subSetUPertinencia[][] = new double[2][length];
		
		for(int i = 0; i< length; i++){
			subSetUPertinencia[0][i] = i ;
		    
			String u1Class= subSetU1.instance(i).stringValue(subSetU1.classAttribute());
			String u2Class= subSetU1.instance(i).stringValue(subSetU1.classAttribute());
			
			// Caso eles concordarem entre eles somar valor de pertinência. 
			if ( u1Class.equals(u2Class)){
				subSetUPertinencia[1][i] = subSetU1Pertinencia[1][i]+subSetU2Pertinencia[1][i];
			}else{
				subSetUPertinencia[1][i] = 0;
			}
		}
						
		// Ordernando subsetUPertinencia 
		QuickSort sorter = new QuickSort();
		sorter.sort(subSetUPertinencia, length);
	    /** /	sorter.printArray(subSetUPertinencia,length);  /* Imprimir vetor ordenado */
		
	 	/*
	 	 *  CLASSIFICANDO SET U
	 	 */
	
		// Definindo a porcetagem que será selecionada para o conjunto L
		if (set_porcent == 0){
				set_porcent = (length*percentMe)/100 ;				
		} 
		
		// Redefinindo a porcetagem, ciclo final do algoritmo.
		if (setU.numInstances()<set_porcent){
			set_porcent = setU.numInstances();
		}
					
		
		// A seleção é feita de tráz para frente pois queremos as intâncias com o maior grau de pertinência
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
	
	private Evaluation crossValidationTeste(Instances data,Classifier cls) throws Exception{
		
		// randomize data
	    Random rand = new Random(seed);
	    Instances randData = new Instances(data);
	    randData.randomize(rand);
	    if (randData.classAttribute().isNominal())
	      randData.stratify(folds);

	    // perform cross-validation
	    Evaluation eval = new Evaluation(randData);
	    for (int n = 0; n < folds; n++) {
	      Instances train = randData.trainCV(folds, n);
	      Instances test = randData.testCV(folds, n);
	      // the above code is used by the StratifiedRemoveFolds filter, the
	      // code below by the Explorer/Experimenter:
	      // Instances train = randData.trainCV(folds, n, rand);

	      // build and evaluate classifier
	      Classifier clsCopy = (Classifier)new SerializedObject(cls).getObject();	      
	      
	      clsCopy.buildClassifier(train);
	      eval.evaluateModel(clsCopy, test);
	      	      
	    }
	    
		return eval;		
	}	
	
	/**
	 * Método para imprimir informações do classificador e matriz de confusão.
	 * 
	 * @param test Evaluation - Avaliação disponivel depois de gerar o classificador.
	 */
	private void imprimeResultadoH(Evaluation test) {
	
		String strSummary="";
		
		if (modTraining == 0){		
			try {
				strSummary = test.toSummaryString()+"\n"+test.toMatrixString();
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}		
		}		
		
		if (modTraining == 1){		
			try {
				strSummary = test.toSummaryString("=== " + folds + "-fold Cross-validation ===",false)+"\n"+test.toMatrixString();
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}		
		}
		
		m_OutText.setText(m_OutText.getText()+"\n"+strSummary+"\n");		
		
	}

	/**
	 * Método destinado a setar  como missing o valor da classe de determinada porcentagem
	 * de instâncias do conjunto de dados escolhido pelo usuário.
	 */
	private Instances testeSelection(int porcentagem){
		
		//Copiando todos os registros do cojunto original para meu conjunto de testes.
		Instances trainingSetMissing = new Instances(setOriginal);
		
		// Capturando a representação da quantidade de registros dada a porcentagem. 
		int registros = (int) ((trainingSetMissing.numInstances()*porcentagem)/100);
		
		/*O laço de repetição vai diminuindo a quantidade de registros até a porcentagem presente do grupo seja exatamente
		 a que o usuário determinou */
		
		// Detalhe importante, registros é o numero total de instâncias que deveram ter suas
		// classes setadas como missing.
		// Registro no singular é  um index selecionado de maneira aleatória para setar no 
		// conjunto  de dados uma instância com a classe missing.
		Random random = new Random();		
		int registro;
		
		// limpando m_outText
		m_OutText.setText("");
		
		if (printFinalTeste){
			out("\n Selecting a group of instances of the original data set. \n "
					+ "================================================================================");
			
			out(" Full instances : "+trainingSetMissing.numInstances()+" Full selected instances without label:"+registros);
			out(" Starting random selection...");
		}
		
		ArrayList<Integer> saiu= new ArrayList<Integer>();
		boolean check;	
		
		while (registros>0 ){
			
			registro = random.nextInt(trainingSetMissing.numInstances());
			
			// Certificando que o random sempre escolha alguém diferente.
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
		
		if (printFinalTeste){
			out(" Success! Set of test data was selected successfully. \n "
				+ "end selection \n"
				+ "================================================================================ \n");
		}
									
		return trainingSetMissing;
	}
			
	/**
	 *  Método destinado para a realizaçãoo do teste final utilizando o cojunto gerado a partir do cojunto de 
	 *  dados trainingSetMissing criado a partir da execução do método testeSelection. 
	 */
	private void testeResult(Instances trainingSetMissing){
		
		double classe = 0;
				
		out("\nTesting the results of the classifiers, final test.\n"
				+ "This test compares the label indicated by the classifiers with the actual value of the original set. \n"
					+ "================================================================================\n");											
		
		int corretamenteClassificadosH1   = 0,
			corretamenteClassificadosH2   = 0,
	        igualdadeEntreClassificadores = 0;
		
		double N11 = 0; //é o número de padrões onde ambos os classificadores erram;
		double N00 = 0; //é o número de padrões onde ambos os classificadores acertam;
		double N10 = 0; //é o número de padrões onde o primeiro classificador erra e o segunda acerta;
		double N01 = 0; //é o número de padrões onde o primeiro classificador erra e o segunda acerta;
		double entropia= 0; //Entropia
		
		
		double Qestatistico = 0;
		double MedidadeDesacordo = 0;
		double MedidadeDuploFalso = 0;
			
		//Classificando com o Classificador 1
		for(Instance registro: subSetL1){			
			
			try {
				classe = classifierH1.classifyInstance(registro);
			} 
			catch (Exception e) {								
					e.printStackTrace();
					return;
				}
			
			//Capturando o valor da classe indicada pelo classificador
	        String valor = registro.attribute(registro.classIndex()).value((int) classe);
	        //Atribuindo a instância o valor da sua classe
	        registro.setValue(registro.classIndex(),valor);
	        	       
		}					
		
		//Classificando com o Classificador 2
		for(Instance registro: subSetL2){			
			
			try {
				classe = classifierH2.classifyInstance(registro);
			} 
			catch (Exception e) {								
					e.printStackTrace();
					return;
				}
			
			// Capturando o valor da classe indicada pelo classificador
	        String valor = registro.attribute(registro.classIndex()).value((int) classe);
	        // Atribuindo a instância o valor da sua classe
	        registro.setValue(registro.classIndex(),valor);	        			
		}		
		
				    		
		setOriginal.setClassIndex(classIndex);
		
		for(int i=0; i<setOriginal.numInstances();i++){	
			
			boolean hitH1 = false;
			boolean hitH2 = false;
			
			    /** Ambos os classificadores erram */
			if(!setOriginal.instance(i).stringValue(setOriginal.classIndex())
					.equals(subSetL1.instance(i).stringValue(subSetL1.classIndex())) ){				
					if(!setOriginal.instance(i).stringValue(setOriginal.classIndex())
							.equals(subSetL2.instance(i).stringValue(subSetL2.classIndex())) ){
						N11++;
			     	}
			 }
			
		    /** Ambos os classificadores acertam */
			if(setOriginal.instance(i).stringValue(setOriginal.classIndex())
					.equals(subSetL1.instance(i).stringValue(subSetL1.classIndex())) ){				
					if(setOriginal.instance(i).stringValue(setOriginal.classIndex())
							.equals(subSetL2.instance(i).stringValue(subSetL2.classIndex())) ){
						N00++;
			     	}
			 }

			 /** Primeiro erra segundo acerta */
			if(!setOriginal.instance(i).stringValue(setOriginal.classIndex())
					.equals(subSetL1.instance(i).stringValue(subSetL1.classIndex())) ){				
					if(setOriginal.instance(i).stringValue(setOriginal.classIndex())
							.equals(subSetL2.instance(i).stringValue(subSetL2.classIndex())) ){
						N10++;
			     	}
			 }
			
			 /** Primeiro acerta segundo erra */
			if(setOriginal.instance(i).stringValue(setOriginal.classIndex())
					.equals(subSetL1.instance(i).stringValue(subSetL1.classIndex())) ){				
					if(!setOriginal.instance(i).stringValue(setOriginal.classIndex())
							.equals(subSetL2.instance(i).stringValue(subSetL2.classIndex())) ){
						N01++;
			     	}
			 }
				
					  
			
				//Verificando se a classe sugerida pelo classificador 1 condiz com o valor verdadeiro da classe.
				if( setOriginal.instance(i).stringValue(setOriginal.classIndex())
						.equals(subSetL1.instance(i).stringValue(subSetL1.classIndex())) ){
					corretamenteClassificadosH1++;
					hitH1 = true;
				}
				
				//Verificando se a classe sugerida pelo classificador 2 condiz com o valor verdadeiro da classe.
				if( setOriginal.instance(i).stringValue(setOriginal.classIndex())
						.equals(subSetL2.instance(i).stringValue(subSetL2.classIndex())) ){
					corretamenteClassificadosH2++;
					hitH2 = true;
				}
				
				//Verificando se a classe sugerida pelo classificador 1 � a mesma do classificador 2. 
				if( subSetL1.instance(i).stringValue(subSetL1.classIndex())
						.equals(subSetL2.instance(i).stringValue(subSetL2.classIndex())) ){
					igualdadeEntreClassificadores++;
				}		
				
				if (hitH1 == true &&  hitH2 == true){
					// l(Zm) = 2 , 2 - l(Zm) => 2 -2 = 0 , escolhido 0 
				}else{
					if (hitH1 == true || hitH2 == true){
						entropia += 1.0;
					}
				}
				
		}
		
		
		if (entropia != 0) {
			entropia = entropia/subSetL1.numInstances();
		}
		
		Qestatistico = ((N11*N00)-(N01*N10))/((N11*N00)+(N01*N10));
		MedidadeDesacordo = (N10+N01)/(N11+N00+N01+N10);
		MedidadeDuploFalso = N00/(N11+N00+N01+N10);
		
		NumberFormat formatter = new DecimalFormat("#0.0000"); 
			 									
		out("\n Q-Estatístico : "+formatter.format(Qestatistico));
		out(" Medida de Desacordo : "+formatter.format(MedidadeDesacordo));
		out(" Medida de Duplo-Falso : "+formatter.format(MedidadeDuploFalso));
		out(" Entropy : "+formatter.format(entropia)+"\n");
	
		out("Correctly Classified Instances     "+formatter.format(((100.0*((double)corretamenteClassificadosH1))/((double)trainingSetMissing.numInstances())))+" % \n"
				+ "		Correctly classified of H1 :"+corretamenteClassificadosH1+"\n"
				+ "		Correctly classified of H2 :"+corretamenteClassificadosH2+"\n"
				+ "		Equality between the classifiers :"+igualdadeEntreClassificadores+"\n"
				+ "		Both classifiers missed :"+N11 +"\n"
				+ "		First missed, and the second hit :"+N10+"\n"
				+ "		Second missed, and the first hit :"+N01);
		
		out("\n"
				+ "================================================================================ \n");
		
	}
	
	
	/**
	 *  Método responsável por verificar se todos os atributos necessários para o 
	 *  funcionamento dessa classe estão com seus valores setados. 
	 * 
	 * @return boolean - Caso esteja tudo bem retorna true, caso não retorna false.
	 */
	private boolean validarEntradas(){
		
		if (  setOriginal == null ){
			System.err.println(" Não é possível continuar, para o CoTraning ser iniciado"
					+ " É preciso atribuir o valor do atributo setOriginal. \n Tente utilizar"
					+ " o método setSetOriginal(Intances) logo após o construtor da classe CoTraning.");
			return false;
		}
		
		if (  algoClassificacao1 == null ){
			System.err.println(" Não é possível continuar, para o CoTraning ser iniciado"
					+ " É preciso atribuir o valor do atributo algoClassificacao1. \n Tente utilizar"
					+ " o método setAlgoClassificacao1(Classifier) logo após o construtor da classe CoTraning.");
			return false;
		}
		
		if (  algoClassificacao2 == null ){
			System.err.println(" Não é possível continuar, para o CoTraning ser iniciado"
					+ " É preciso atribuir o valor do atributo algoClassificacao2. \n Tente utilizar"
					+ " o método setAlgoClassificacao2(Classifier) logo após o construtor da classe CoTraning.");
			return false;
		}      
		
		if (  m_OutText == null ){
			System.err.println(" Não é possível continuar, para o CoTraning ser iniciado"
					+ " É preciso atribuir o valor do atributo m_OutText. \n Tente utilizar"
					+ " o método setM_OutText(JTextPane) logo após o construtor da classe CoTraning.");
			return false;
		}  
		
		if (  listaAtributosUm == null ){
			System.err.println(" Não é possível continuar, para o CoTraning ser iniciado"
					+ " É preciso atribuir o valor do atributo listaAtributosUm. \n Tente utilizar"
					+ " o método setListaAtributosUm(ArrayList<Boolean>) logo após o construtor da classe CoTraning.");
			return false;
		}  
		      
		if (  listaAtributosDois == null ){
			System.err.println(" Não é possível continuar, para o CoTraning ser iniciado"
					+ " É preciso atribuir o valor do atributo listaAtributosDois. \n Tente utilizar"
					+ " o método setListaAtributosDois(ArrayList<Boolean>) logo após o construtor da classe CoTraning.");
			return false;
		}  
								
		
		return true;
	}
	
	
	/**
	 * @return the classifierH1
	 */
	public Classifier getClassifierH1() {
		return classifierH1;
	}

	/**
	 * @param classifierH1 the classifierH1 to set
	 */
	public void setClassifierH1(Classifier classifierH1) {
		this.classifierH1 = classifierH1;
	}

	/**
	 * @return the classifierH2
	 */
	public Classifier getClassifierH2() {
		return classifierH2;
	}

	/**
	 * @param classifierH2 the classifierH2 to set
	 */
	public void setClassifierH2(Classifier classifierH2) {
		this.classifierH2 = classifierH2;
	}


	public void setSetOriginal(Instances setOriginal) {
		this.setOriginal = setOriginal;
	}

	public Classifier getAlgoClassificacao1() {
		return algoClassificacao1;
	}

	public void setAlgoClassificacao1(Classifier algoClassificacao1) {
		this.algoClassificacao1 = algoClassificacao1;
	}

	public Classifier getAlgoClassificacao2() {
		return algoClassificacao2;
	}

	public void setAlgoClassificacao2(Classifier algoClassificacao2) {
		this.algoClassificacao2 = algoClassificacao2;
	}

	public int getClassIndex() {
		return classIndex;
	}

	public void setClassIndex(int classIndex) {
		this.classIndex = classIndex;
	}


	public void setM_OutText(JTextPane m_OutText) {
		this.m_OutText = m_OutText;
	}

	public ArrayList<Boolean> getListaAtributosUm() {
		return listaAtributosUm;
	}

	public void setListaAtributosUm(ArrayList<Boolean> listaAtributosUm) {
		this.listaAtributosUm = listaAtributosUm;
	}

	public ArrayList<Boolean> getListaAtributosDois() {
		return listaAtributosDois;
	}

	public void setListaAtributosDois(ArrayList<Boolean> listaAtributosDois) {
		this.listaAtributosDois = listaAtributosDois;
	}
	
	public void setSeed(int seed){
		 this.seed= seed ;
	}	
	
	public void setFolds(int folds){
		this.folds = folds;
	}
	
	public void setPercentageUnlabeled(int percentageUnlabeled){
		this.percentageUnlabeled = percentageUnlabeled;
	}

	public void setPrintFinalTeste(boolean printFinalTeste){	
		this.printFinalTeste = printFinalTeste;
	}
	
	public void out(String text){
		m_OutText.setText(m_OutText.getText()+text+"\n");
	}
	
	public void setModTraining(int i){
		modTraining = i;
	}
	
	// Porcentegem de incremento por ciclo dos melhores exemplos
	public void setPercentMe(int i){
		percentMe = i ;
	}
	
	public void setM_Log( Logger m_Log){
		this.m_Log= m_Log ;
	}
	
	public void setAssociator(Associator a){
		associator = a;
	}
	
}