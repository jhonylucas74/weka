package weka.classifiers.semisupervised;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import javax.swing.JTextPane;

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
import weka.core.SerializedObject;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.learning.semisupervised.InstanceMeta;

public class SelfTraining extends AbstractClassifier 
implements OptionHandler, WeightedInstancesHandler, TechnicalInformationHandler{

	public SelfTraining(){
		classifierBase = new IBk();
		clustererBase =  new EM();
		folds = 10;
		percentUnlabeledInstances = 10;
		incorporatePerCycle = 10;
	}
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -6980995715281717775L;

	// Vari�veis utilizadas localmente
	private Instances setOriginal;      //Conjunto original recebido da interface
	
	private Instances setL; 			//Sub-conjunto de dados rotulados
	
	private Instances setU; 			//Sub-conjunto de dados n������o-rotulados
	
	private Instances subSetL; 			//Sub-conjunto L' que ir������ conter todos os exemplos de U rotulados por H	
	
	private Evaluation eTest;		
	
	private int seed = 30;
	
	private int classIndex;
	
	public static JTextPane m_OutText;
	
	private boolean printFinalTeste = true;
	
	// Vari�veis necess�rias para construir o classificador
	
	protected Classifier classifierBase;
	
	protected boolean ComparativeTestResult = true;
	
	protected int folds;
	
	protected int percentUnlabeledInstances;
	
	protected int incorporatePerCycle;
	
	protected Clusterer clustererBase;
	
	protected boolean selectionWithClustering;

	protected boolean crossValidate;
	
	protected boolean relativeIncorporation;
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		runClassifier(new SelfTraining(), args);
	}
	
	public double classifyInstance(Instance instance) throws Exception {

		double classIndex = getClassifierBase().classifyInstance(instance);        	
		String valor = instance.classAttribute().value((int) classIndex);
		instance.setValue(setL.classAttribute(), valor);
		
		return classIndex;
		  
	}
	
	@Override
	public void buildClassifier(Instances data) throws Exception {
		
		this.setOriginal = data;
		this.classIndex = setOriginal.classIndex();
		
		Instances trainingSetMissing = testSelection(percentUnlabeledInstances);
		separatingSet(trainingSetMissing);
		
		if(isSelectionWithClustering()){
			try {
				buildClusterer();
			} catch (Exception e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
		}

		int hits = (int) Math.round( 1 / incorporatePerCycle );
		
		double quantidade = Math.ceil( setU.numInstances() * incorporatePerCycle );
		
		buildClassifierH();

		try {
			classificaU();// gerando U' a partir do classificador H
		} catch (Exception e) {
			e.printStackTrace();
		} 				

		if(isSelectionWithClustering()){
			try {
				cluster(incorporatePerCycle, relativeIncorporation);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		else{
			for(int hit = hits; hit > 0; hit--) {
				try {
					if(quantidade > setU.numInstances())
						quantidade = setU.numInstances();
					melhoresExemplos((int)quantidade);
				} catch (Exception e) {
					e.printStackTrace();
				}							// Seleciona as melhores int����ncias de L' e as adiciona em L
				
				buildClassifierH();		// Passando o algoritmo de classifi��������o		
			}
		}
		
        /*
         *  Testando conjunto de dados utilizando cross - validation        
         */
        Evaluation eval = null;
        
        if (isCrossValidate()) {
		        try {
		        	eval = crossValidate(setOriginal, classifierBase);
				} catch (Exception e) {		
					e.printStackTrace();
				}
        } else {
        	eval = eTest;
        }
	}

	public void buildClassifierH(){
		
		eTest = null;
			
		try {
			setL.setClassIndex(classIndex);
			getClassifierBase().buildClassifier(setL); 		// utilizando o conjunto L para treinamento
			eTest = new Evaluation(setL);
			eTest.evaluateModel(getClassifierBase(), setL); // testando o modelo gerado a partir do treinamento
		} catch (Exception e) {
			e.printStackTrace();
			
		}
	}
	
	/** Escolhe as melhores int������ncias dos exemplos que foram rotulados (U') 
	 * 	para serem adicionados ao conjunto de exemplos rotulados (L)
	 * @throws Exception 
	 *  
	 */
	public void melhoresExemplos(int quantidade) throws Exception {
		
		if(setU.size() == 0) //se n������o houver exemplos para incorporar... retorne
			return;

		ArrayList<InstanceMeta> listMeta = new ArrayList<InstanceMeta>();	//cole������������o de objetos com meta-informa������������es sobre as inst������ncias
		InstanceMeta instanceMeta; 					//objeto com meta-informa������������es de cada inst������ncia
		
		//la������o para construir a lista de meta-informa������������es
		for (int i = 0; i < subSetL.size(); i++) {
			double distClass[] = getClassifierBase().distributionForInstance(subSetL.instance(i));
			
			instanceMeta = new InstanceMeta();			//instanciando o objeto
			instanceMeta.setIntanceIndex(i);					//recebe o ������ndice da inst������ncia
			instanceMeta.setMaxClassValue(distClass); 	//recebe a distribui������������o pelos grupos da dada instancia

			listMeta.add(instanceMeta);					//adiciona o objeto criado ������ lista de meta-dados
		}

		listMeta = quicksort(listMeta);		//classifica a lista de meta-dados pelo grau de pertinencia da classe
		
		int index = 0;			//indice da inst������ncia
		int hits = quantidade;	//quantidade de exemplos que ser������o incorporados por vez
		
		ArrayList<Integer> lista = new ArrayList<Integer>();	//lista que conter������ os indices das instancias para auxiliar na remo������������o
		
		while (hits > 0) {
			instanceMeta = listMeta.get(index);
			
			lista.add( instanceMeta.getIntanceIndex() );
			//mapeando a instancia no subSetL a partir de seu ������ndice e adicionando-a no subSetL
			setL.add( subSetL.instance(instanceMeta.getIntanceIndex()) );
			
			index++;
			hits--;
		}
		
		/*	ordena os ������ndices decrescentemente para que a exclus������o de uma instancia n������o modifique 
		 * 	a ordem das demais instancias que tamb������m ser������o exclu������das
		 */
		Collections.sort(lista, Collections.reverseOrder());
		
		for (int i = 0; i < lista.size(); i++){
			if(setU.size() > lista.get(i))
				setU.delete(lista.get(i)); 			//apagando do conjunto U as inst������ncias que j������ foram incorporadas
		}
	}
	

	public void buildClusterer() throws Exception {
		//Cria��������o do agrupador
		setClustererBase(clustererBase);
		// atribuindo o algoritmo a classe respons����vel por agrupar
		getClustererBase().buildClusterer(setU);				// construindo agrupador...	
	}
	
	/**
	 * Aplica t����cnicas de agrupamento para otimizar a classifica��������o dos exemplos n����o-rotulados.
	 * Este m����todo incorpora ���� base rotulada uma por��������o de instancias por vez, 
	 * onde essa quantidade deve fornecida em percentual.
	 * @param percent o percentual de exemplos/grupo | o percentual de exemplos do conjunto
	 * @param type 
	 * @throws Exception
	 */
	public void cluster(double percent, boolean relativeIncorporation) throws Exception {
		
    	int numClusters = setL.numClasses();			 	//armazena a quantidade de grupos
    	int[] qtdExemplosPorGrupo = new int[numClusters];	//armazena a quantidade de instancias/grupo
    	int countUnlabeledInstances = setU.size(); 			//armazena a quantidade de instancias n����o-rotuladas 
    	
    	if(!isRelativeIncorporation()) {	// retire x elementos de cada grupo (balanceado)
    		int qtd = (int) Math.ceil(percent * countUnlabeledInstances / numClusters);
    		
    		for (int i = 0; i < numClusters; i++){
	    		if(qtd < 1)
    				qtdExemplosPorGrupo[i] = 1;
    			qtdExemplosPorGrupo[i] = qtd;
    		}
    	}
    	
    	//gera novo classificador com base nos exemplos adicionados ao conjunto de dados rotulados 
    	buildClassifierH();
    	try {
			classificaU();
		} catch (Exception e) {
			e.printStackTrace();
		}	
    	
    	//fa��a... enquanto houver elementos a serem incorporados
    	while (!subSetL.isEmpty()) {
    		//define um grupo para cada instancia do conjunto j� rotulado
    		
    		ArrayList<InstanceMeta> completeSet = clusterLabeledInstances();
    		
        	//incluindo cada instancia em seu respectivo grupo 
    		//ordenando os grupos com base no grau de pertinencia da classe dos elementos
        	//adicionando o grupo (ordenado) ao conjunto de grupos
    		ArrayList<ArrayList> clusters = new ArrayList<>();	//cole��������o de grupos
        	for (int i = 0; i < numClusters; i++) {
        		ArrayList<InstanceMeta> cluster; 			//grupo individual
        		
        		cluster = createCluster(completeSet, i);	//cria novo grupo a partir do conjunto completo + ����ndice do grupo 
        		clusters.add(quicksort(cluster));
        		
    			//retire x% dos elementos de cada grupo (relativo)
    			//preenche o vetor com a quantidade de exemplos para este grupo
    			if( (isRelativeIncorporation()) && (setU.size() == countUnlabeledInstances) ){
    				
    				qtdExemplosPorGrupo[i] = (int) Math.round( percent * cluster.size() );
    				
    				if (qtdExemplosPorGrupo[i] < 1 && cluster.size() > 0)
    					qtdExemplosPorGrupo[i] = 1;
    			}
        	}
        	
        	//teste
        	printResults(clusters);
        	
            //incorporando elementos de cada grupo ���� base rotulada
            for (int i = 0; i < numClusters; i++) {
            	if (! (clusters.get(i).size() > 0))
            		return;
        	
            	if(qtdExemplosPorGrupo[i] > clusters.get(i).size() )
        			qtdExemplosPorGrupo[i] = clusters.get(i).size();
            	
            	incorporateByTime(clusters.get(i), qtdExemplosPorGrupo[i]);
            }
            
            //gera novo classificador com base nos exemplos adicionados ao conjunto de dados rotulados 
            buildClassifierH();
        	try {
    			classificaU();
    		} catch (Exception e) {
    			e.printStackTrace();
    		}	
        }
	}
	
	private void printResults(ArrayList<ArrayList> clusters) {
		// TODO Auto-generated method stub
		
	}
	
	/**
	 * Agrupa os exemplos rotulados desconsiderando o valor de seu r����tulo.
	 * @return um conjunto de instancias agrupadas
	 * @throws Exception
	 */
	private ArrayList<InstanceMeta> clusterLabeledInstances() throws Exception {
		int clusterValue;				//grupo da instancia i
		double classValue;				//classe da instancia i
    	double distClass[];				//distribui��������o dos grupos para a da da instancia
    	
    	ArrayList<InstanceMeta> instancesRFull = new ArrayList<InstanceMeta>(); //armazena instanceClass
    	
    	//la����o para adicionar cada instancia (com o valor do r����tulo) a um conjunto
    	for (int i = 0; i < subSetL.size(); i++) {
			classValue = subSetL.instance(i).classValue(); 
			subSetL.instance(i).setClassMissing(); 								//oculta o valor da classe
    		clusterValue = getClustererBase().clusterInstance(subSetL.instance(i));		//agrupa o exemplo
			distClass = getClassifierBase().distributionForInstance(subSetL.instance(i));
    		subSetL.instance(i).setClassValue(classValue); 						//torna a classe vis����vel novamente
        	
        	//armazena os valores do ����ndice, do grupo, e da distribui��������o das classes para a dada instancia
    		InstanceMeta instanceMeta = new InstanceMeta(i, classValue, clusterValue, distClass);	
        	
        	instancesRFull.add(instanceMeta);	// todos os exemplos do conjunto subSetL agrupados
        }
		
		return instancesRFull;
	}
	
	private ArrayList<InstanceMeta> createCluster(ArrayList<InstanceMeta> instancesRFull, int i) {
		ArrayList<InstanceMeta> cluster = new ArrayList<>();	// cria��������o do grupo
    	
		for (int j = 0;  j < instancesRFull.size(); j++) {	// la����o para os exemplos rotulados e agrupados	
    		double clusterValue = instancesRFull.get(j).getClusterValue();
    		
    		if (clusterValue == i)
				cluster.add(instancesRFull.get(j));
		}

    	return cluster;
	}
	
	private void incorporateByTime(ArrayList<InstanceMeta> cluster, int numberSamples) {
		
		if (! ((numberSamples > 0 ) && (cluster.size() > 0)) ) {
			return;
		}
		
		ArrayList<Integer> lista = new ArrayList<Integer>();
		InstanceMeta instance;
		
		for (int i = numberSamples - 1; i >= 0 ; i--) {
			instance = cluster.get(i);
			
			int index = instance.getIntanceIndex();
			
			setL.add( subSetL.get(index) );	// mapeando a instancia pelo seu indice, e incorporando-a
			
			lista.add(index);
		} // fim do for
	
		Collections.sort(lista, Collections.reverseOrder());
				
		for (int i = 0; i < lista.size(); i++){
			if(setU.size() > lista.get(i))
				setU.delete(lista.get(i)); 
		}
		
	}
	
	
	/**
	 * This method sort the input ArrayList using quick sort algorithm.
	 * @param clusterInstancesR the ArrayList.
	 * @return sorted ArrayList of integers.
	 */
	private ArrayList<InstanceMeta> quicksort(ArrayList<InstanceMeta> clusterInstancesR){
		
		if(clusterInstancesR.size() <= 1)
			return clusterInstancesR;
		
		int middle = (int) Math.ceil((double)clusterInstancesR.size() / 2);
		InstanceMeta pivot = clusterInstancesR.get(middle);
 
		ArrayList<InstanceMeta> less = new ArrayList<InstanceMeta>();
		ArrayList<InstanceMeta> greater = new ArrayList<InstanceMeta>();
		
		for (int i = 0; i < clusterInstancesR.size(); i++) {
			if(clusterInstancesR.get(i).getMaxClassValue()
					<= pivot.getMaxClassValue()){
				if(i == middle)
					continue;
				less.add(clusterInstancesR.get(i));
			}
			else{
				greater.add(clusterInstancesR.get(i));
			}
		}
		
		return concatenate(quicksort(greater), pivot, quicksort(less));
	}
	
	/**
	 * Join the less array, pivot integer, and greater array
	 * to single array.
	 * @param less integer ArrayList with values less than pivot.
	 * @param pivot the pivot integer.
	 * @param greater integer ArrayList with values greater than pivot.
	 * @return the integer ArrayList after join.
	 */
	private ArrayList<InstanceMeta> concatenate(ArrayList<InstanceMeta> greater, 
			InstanceMeta pivot, ArrayList<InstanceMeta> less){
		
		ArrayList<InstanceMeta> list = new ArrayList<InstanceMeta>();
		
		for (int i = 0; i < greater.size(); i++) {
			list.add(greater.get(i));
		}
		list.add(pivot);
		for (int i = 0; i < less.size(); i++) {
			list.add(less.get(i));
		}
		
		return list;
	}
	
	private Instances testSelection(int porcentagem){
		
		//Copiando todos os registros do cojunto original para meu conjunto de testes.
		Instances trainingSetMissing = new Instances(setOriginal);
		
		// Capturando a representa��������o da quantidade de registros dada a porcentagem. 
		int registros = (int) ((trainingSetMissing.numInstances()*porcentagem)/100);
		
		/*O la����o de repeti��������o vai diminuindo a quantidade de registros at���� a porcentagem presente do grupo seja exatamente
		 a que o usu����rio determinou */
		
		// Detalhe importante, registros ���� o numero total de inst����ncias que deveram ter suas
		// classes setadas como missing.
		// Registro no singular ����  um index selecionado de maneira aleat����ria para setar no 
		// conjunto  de dados uma inst����ncia com a classe missing.
		Random random = new Random();		
		int registro;
		
		// limpando m_outText
		//m_OutText.setText("");
		
		if (printFinalTeste) {
			System.out.println("\n Selecting a group of instances of the original data set. \n "
					+ "================================================================================");
			
			System.out.println(" Full instances : "+trainingSetMissing.numInstances()+" Full selected instances without label:"+registros);
			System.out.println(" Starting random selection...");
		}
		
		ArrayList<Integer> saiu= new ArrayList<Integer>();
		boolean check;	
		
		while (registros>0 ){
			
			registro = random.nextInt(trainingSetMissing.numInstances());
			
			// Certificando que o random sempre escolha algu����m diferente.
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
			System.out.println(" Success! Set of test data was selected successfully. \n "
				+ "end selection \n"
				+ "================================================================================ \n");
		}
									
		return trainingSetMissing;
	}
	
	/**  Separa o conjunto de instancias em dois sub-conjuntos:
	 *   Sub-conjunto L: inst��ncias r��tuladas
	 *   Sub-conjunto U: inst��ncias n��o r��tuladas
	 *  
	 *  @param setOriginal Instances - Conjunto de dados original
	 */
	public void separatingSet(Instances setOriginal) {

		setOriginal.setClassIndex(setOriginal.numAttributes() - 1);
		
		//Cria o sub-conjunto L
		setL = new Instances(setOriginal); 					// cria uma c��pia do conjunto original
		setL.setRelationName("Set L");     					// nomeia o sub-conjunto L
		setL.setClassIndex(setL.numAttributes() - 1); 		// define o atributo-classe
		setL.deleteWithMissingClass(); 						// elimina todos os elementos sem-r��tulos
		
		//Cria o sub-conjunto U         
		setU = new Instances(setOriginal); 					//cria uma c��pia do conjunto original
		setU.setRelationName("Set U"); 	   					//nomeia o sub-conjunto U
		setU.setClassIndex( -1 );							//define conjunto sem classe
		// percorre todo o sub-conjunto U para eliminar todos os elementos com r��tulos
		int numInstances = setU.numInstances();
		for( int i = (numInstances-1); i > -1 ; i-- ) {
			if( !setOriginal.instance(i).classIsMissing() )
				setU.delete(i);
		}
	}
	
	private Evaluation crossValidate(Instances data, Classifier cls) throws Exception{
		
		// randomize data
		data.setClassIndex(classIndex);
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
	
	/** Classifica o conjunto de dados n������o-rotulados com base no classificador H
	 * 
	 * @throws Exception 
	 */
	public void classificaU() throws Exception {
		subSetL = new Instances(setU); // O sub conjunto L' tem todos os elementos do conjunto U
		//Renomeando o sub conjunto L'
		subSetL.setRelationName("Sub set L'");              
		//Definindo o atributo classe.
		subSetL.setClassIndex(classIndex); 
	
		//Classificando com H
		for(int i=0; i < subSetL.numInstances(); i++) {
			double classe = getClassifierBase().classifyInstance(subSetL.instance(i));        	
			String valor = subSetL.instance(i).attribute(subSetL.numAttributes() - 1).value((int) classe);
			subSetL.instance(i).setValue(subSetL.numAttributes() - 1,valor);
		}		
	}
	
	@Override
	public TechnicalInformation getTechnicalInformation() {
		// TODO Auto-generated method stub
		return null;
	}
	

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
	*  For instances selection with clustering. </pre>
	* 
	* <pre> -R
	*  For relative incorporation mode. </pre>
	* 
	* <pre> -I
	*  Percent value to incorporate per cycle. </pre>
	* 
	* <pre> -F 
	*  Percent value to set for folds at cross-validation.
	* (Default = 10)</pre>
	* 
	* <pre> -X
	*  Select cross-validate for training data.
	* </pre>
	* 
	* <pre> -C
	*  Select the classifier base algorithm to use (default: weka.classifiers.lazy.IBk).
	* </pre>
	*
	* <pre> -G
	*  The cluster base algorithm to use (default: weka.clusterers.EM).
	* </pre>
	* 
	<!-- options-end -->
	*/
	
	public Enumeration listOptions() {
	
		Vector newVector = new Vector(4);
	
		newVector.addElement(new Option(
				"\tSelect relative incorporation mode.",
				"R", 0,"-R"));
	
		newVector.addElement(new Option(
				"\tSelect cross-validate for training data.",
				"X", 0,"-X"));
				
		newVector.addElement(new Option(
				"\tFor instances selection with clustering.",
				"S", 0,"-S"));		
			  
		newVector.addElement(new Option(
				"\tPercent value to set for folds at cross-validation.",
				"F", 0,"-F"));
		
		newVector.addElement(new Option(
				"\tPercent value to incorporate per cycle.",
				"I", 0,"-I"));
		
		newVector.addElement(new Option(
				"\tPercent value of unlabeled instances.",
				"U", 0,"-U"));
		
		newVector.addElement(new Option(
				"\tSelect the classifier base algorithm to use (default: weka.classifiers.lazy.IBk).",
				"C", 0,"-C"));
		
		newVector.addElement(new Option(
				"\tThe cluster base algorithm to use (default: weka.clusterers.EM).",
				"G", 0,"-G"));
				
		return newVector.elements();
	}
	
	public void setOptions(String[] options) throws Exception {
	
		setCrossValidate(Utils.getFlag('X', options));
		setRelativeIncorporation(Utils.getFlag('R', options));
		setSelectionWithClustering(Utils.getFlag('S', options));
		String folds = Utils.getOption('F', options);
		if (folds.length() != 0) {
			setFolds(Integer.parseInt(folds));
		} else {
			setFolds(10);
		}
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
		String classifierBase = Utils.getOption('C', options);
		if (classifierBase.length() != 0) {
			String classifierBaseS[] = Utils.splitOptions(classifierBase);
			if(classifierBaseS.length == 0) { 
			throw new Exception("Invalid classifier algorithm " +
								"specification string."); 
		  }
		  String className = classifierBaseS[0];
		  classifierBaseS[0] = "";

		  setClassifierBase( (Classifier)
					  Utils.forName( Classifier.class, 
									 className, 
									 classifierBaseS)
									);
		} else {
			setClassifierBase(new IBk());
		}
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
	  
	/**
	* Gets the current settings of IBk.
	*
	* @return an array of strings suitable for passing to setOptions()
	*/
	public String [] getOptions() {
	
		String [] options = new String [11];
		int current = 0;
		
		if (isRelativeIncorporation()) {
		  options[current++] = "-R";
		}
		if (isCrossValidate()) {
		  options[current++] = "-X";
		}
		if (isSelectionWithClustering()) {
		  options[current++] = "-S";
		}
		options[current++] = "-F"; options[current++] = "" + getFolds();
		options[current++] = "-I"; options[current++] = "" + getIncorporatePerCycle();
		options[current++] = "-U"; options[current++] = "" + getPercentUnlabeledInstances();
		options[current++] = classifierBase.getClass().getName();
		options[current++] = clustererBase.getClass().getName();
		
		while (current < options.length) {
		  options[current++] = "";
		}
		
		return options;
	}
	
	/**
	 * @return the eTest
	 */
	public Evaluation geteTest() {
		return eTest;
	}

	/**
	 * @param eTest the eTest to set
	 */
	public void seteTest(Evaluation eTest) {
		this.eTest = eTest;
	}

	/**
	 * @return the classifierBase
	 */
	public Classifier getClassifierBase() {
		return classifierBase;
	}

	/**
	 * @param classifierBase the classifierBase to set
	 */
	public void setClassifierBase(Classifier classifierBase) {
		this.classifierBase = classifierBase;
	}

	/**
	 * @return the selectionWithClustering
	 */
	public boolean isSelectionWithClustering() {
		return selectionWithClustering;
	}

	/**
	 * @param selectionWithClustering the selectionWithClustering to set
	 */
	public void setSelectionWithClustering(boolean selectionWithClustering) {
		this.selectionWithClustering = selectionWithClustering;
	}

	/**
	 * @return the crossValidate
	 */
	public boolean isCrossValidate() {
		return crossValidate;
	}

	/**
	 * @param crossValidate the crossValidate to set
	 */
	public void setCrossValidate(boolean crossValidate) {
		this.crossValidate = crossValidate;
	}

	public Clusterer getClustererBase() {
		return clustererBase;
	}

	public void setClustererBase(Clusterer clustererBase) {
		this.clustererBase = clustererBase;
	}

	public Instances getSetOriginal() {
		return setOriginal;
	}

	public void setSetOriginal(Instances setOriginal) {
		this.setOriginal = setOriginal;
	}
	
	public int getFolds() {
		return folds;
	}

	public void setFolds(int folds) {
		this.folds = folds;
	}

	public int getIncorporatePerCycle() {
		return incorporatePerCycle;
	}

	public void setIncorporatePerCycle(int incorporatePerCycle) {
		this.incorporatePerCycle = incorporatePerCycle;
	}

	/**
	 * @return the relativeIncorporation
	 */
	public boolean isRelativeIncorporation() {
		return relativeIncorporation;
	}

	/**
	 * @param relativeIncorporation the relativeIncorporation to set
	 */
	public void setRelativeIncorporation(boolean relativeIncorporation) {
		this.relativeIncorporation = relativeIncorporation;
	}

	public int getPercentUnlabeledInstances() {
		return percentUnlabeledInstances;
	}

	public void setPercentUnlabeledInstances(int unlabeledInstances) {
		this.percentUnlabeledInstances = unlabeledInstances;
	}

}
