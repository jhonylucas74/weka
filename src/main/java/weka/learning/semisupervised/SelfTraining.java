package weka.learning.semisupervised;

/**
 * Implementaï¿½ï¿½o do mï¿½todo semi-supervisionado Self-Training.
 * Segue comentado com devidas observaï¿½ï¿½es; 
 */

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

import javax.swing.JTextPane;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.CostMatrix;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.output.prediction.AbstractOutput;
import weka.clusterers.Clusterer;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializedObject;
import weka.gui.CostMatrixEditor;
import weka.gui.GenericObjectEditor;
import weka.gui.Logger;
import weka.gui.explorer.ClassifierErrorsPlotInstances;
import weka.gui.explorer.ExplorerDefaults;
import weka.gui.explorer.SelfTrainingPanel;

public class SelfTraining extends Thread {

	private Instances setOriginal; // Conjunto original recebido da interface
	private Classifier algoClassificacao; // Algortmo de classificaï¿½ï¿½o
											// determinado pelo usï¿½ario.
	private int classIndex;
	public static JTextPane m_OutText;
	private int modoTreino = -1;

	private Instances setL; // Sub-conjunto de dados rotulados
	private Instances setU; // Sub-conjunto de dados nï¿½o-rotulados
	private Classifier classifierH; // Classificador treinado a partir do
									// cojunto L
	private Instances subSetL; // Sub-conjunto L' que irï¿½ conter todos os
								// exemplos de U rotulados por H
	private Evaluation eTest;
	private Clusterer clusterer;
	private boolean clustererVersion;
	private Clusterer algoAgrupamento;

	private double pctCorreto;

	private double percent;

	/** Atributos que seram utilizados para o modo cross - validation */
	private int seed = 30;
	private int folds = 10;

	/**
	 * Atributo booleano que indicarÃ¡ se serÃ¡ impresso o resultado do teste
	 * final da comparaÃ§Ã£o do conjunto gerado pelos classificadores
	 */
	private boolean printFinalTeste = true;

	private Logger m_Log = new Logger() {

		public void statusMessage(String message) {
			// TODO Auto-generated method stub

		}

		public void logMessage(String message) {
			// TODO Auto-generated method stub

		}
	};

	/**
	 * Porcentagem de instÃ¢ncias que serÃ£o logo no inicio do processo que
	 * terÃ£o seus rÃ³tulos modificados como missing.
	 */
	private int percentageUnlabeled = 10;
	private boolean incorporationMode;

	public SelfTraining(Instances setOriginal, Classifier algoClassificacao,
			int classindex, JTextPane m_OutText, int modoTreino, double percent) {

		this.setOriginal = setOriginal;
		this.algoClassificacao = algoClassificacao;
		this.classIndex = classindex;
		this.m_OutText = m_OutText;
		this.modoTreino = modoTreino;
		this.percent = percent;

		run();
	}

	public SelfTraining() {
	}

	public void run() {

		m_OutText.setText("");

		m_Log.logMessage("Start training  Self-Training");
		m_Log.statusMessage("Start training  Self-Training");

		/** Iniciando Algoritmo do Self-Training */
		// Separando o conjunto de teste em L (rotulados) e U (nao�rotulados)

		Instances trainingSetMissing = testeSelection(percentageUnlabeled);
		separaConjunto(trainingSetMissing);

		if (clustererVersion) {
			try {
				buildClusterer();
			} catch (Exception e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
		}
		// Imprimindo na tela do usï¿½ario
		out("=== Run information === \n" + "Scheme:       "
				+ algoClassificacao.getClass().toString() + "\n"
				+ "Relation:     " + setOriginal.relationName() + "\n"
				+ "Instances:    " + setOriginal.numInstances() + "\n"
				+ "With label:   " + setL.numInstances() + "\n"
				+ "No label:     " + setU.numInstances() + "\n"
				+ "Attributes:   " + setOriginal.numAttributes() + "\n");
		/*
		 * for ( int i=0; i < setOriginal.numAttributes(); i++ ){
		 * m_OutText.setText
		 * (m_OutText.getText()+"              "+setOriginal.attribute
		 * (i).name()+"\n"); }
		 */
		/** Enquanto existir elementos no conjunto de U faça */
		int hits = (int) Math.round(1 / percent);

		double quantidade = Math.ceil(setU.numInstances() * percent);

		criaClassificadorH(algoClassificacao);

		try {
			classificaU();// gerando U' a partir do classificador H
		} catch (Exception e) {
			e.printStackTrace();
		}
		m_Log.statusMessage("Loading training  Self-Training...");

		if (clustererVersion) {
			try {
				// System.out.println("Instancias restantes: "+setU.numInstances());
				cluster(percent, incorporationMode);
			} catch (Exception e) {
				e.printStackTrace();
			}
		} else {
			for (int hit = hits; hit > 0; hit--) {
				try {
					if (quantidade > setU.numInstances())
						quantidade = setU.numInstances();
					melhoresExemplos((int) quantidade);
				} catch (Exception e) {
					e.printStackTrace();
				} // Seleciona as melhores intÃ¢ncias de L' e as adiciona em L

				criaClassificadorH(algoClassificacao); // Passando o algoritmo
														// de classifiÃ§Ã£o
			}
		}

		m_Log.logMessage("End training classifiers");
		m_Log.statusMessage("End training classifiers");

		/*
		 * Testando conjunto de dados utilizando cross - validation
		 */
		Evaluation eval = null;

		if (modoTreino == 2) {
			try {
				eval = crossValidationTeste(setOriginal, algoClassificacao);
			} catch (Exception e) {
				e.printStackTrace();
			}
		} else {
			eval = eTest;
		}

		// m_OutText.setText(m_OutText.getText()+"\n"+"=== Summary ==="+"\n");

		imprimeResultadoH(eTest);

		// Resultado do teste final comparando o real valor das intÃ¢ncias com a
		// classificaÃ§Ã£o dos classifier H gerado.
		if (printFinalTeste) {

			m_Log.logMessage("Start test final ");
			m_Log.statusMessage("Start test final");

			testeResult();
		}

		m_Log.logMessage("End of the execution of Algorithm Self-Training");
		m_Log.statusMessage("OK!");

	}

	private Evaluation crossValidationTeste(Instances data, Classifier cls)
			throws Exception {

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
			Classifier clsCopy = (Classifier) new SerializedObject(cls)
					.getObject();

			clsCopy.buildClassifier(train);
			eval.evaluateModel(clsCopy, test);

		}

		return eval;
	}

	/**
	 * Separa o conjunto de instancias em dois sub-conjuntos: Sub-conjunto L:
	 * instâncias rótuladas Sub-conjunto U: instâncias não rótuladas
	 * 
	 * @param setOriginal
	 *            Instances - Conjunto de dados original
	 */
	public void separaConjunto(Instances setOriginal) {

		setOriginal.setClassIndex(setOriginal.numAttributes() - 1);

		// Cria o sub-conjunto L
		setL = new Instances(setOriginal); // cria uma cópia do conjunto
											// original
		setL.setRelationName("Set L"); // nomeia o sub-conjunto L
		setL.setClassIndex(setL.numAttributes() - 1); // define o
														// atributo-classe
		setL.deleteWithMissingClass(); // elimina todos os elementos sem-rótulos

		// Cria o sub-conjunto U
		setU = new Instances(setOriginal); // cria uma cópia do conjunto
											// original
		setU.setRelationName("Set U"); // nomeia o sub-conjunto U
		setU.setClassIndex(-1); // define conjunto sem classe
		// percorre todo o sub-conjunto U para eliminar todos os elementos com
		// rótulos
		int numInstances = setU.numInstances();
		for (int i = (numInstances - 1); i > -1; i--) {
			if (!setOriginal.instance(i).classIsMissing())
				setU.delete(i);
		}
	}

	/**
	 * Gera o classificador H com base no conjunto dos dados rotulados
	 * 
	 * @param algClassificacao
	 *            Classifier - o algoritmo de classificaï¿½ï¿½o definido pelo
	 *            usuï¿½rio
	 */
	public void criaClassificadorH(Classifier algClassificacao) {
		classifierH = algClassificacao;
		setETest(null);

		try {
			setL.setClassIndex(classIndex);
			classifierH.buildClassifier(setL); // utilizando o conjunto L para
												// treinamento
			setETest(new Evaluation(setL));
			getETest().evaluateModel(classifierH, setL); // testando o modelo
															// gerado a partir
															// do treinamento
		} catch (Exception e) {
			e.printStackTrace();

			m_Log.logMessage(e.getMessage());
			m_Log.statusMessage("Classifier Error - Check the compatibility of the classifier");

			interrupt();
		}
		setPctCorreto(getETest().pctCorrect());
	}

	/**
	 * Classifica o conjunto de dados nï¿½o-rotulados com base no classificador
	 * H
	 * 
	 * @throws Exception
	 */
	public void classificaU() throws Exception {
		subSetL = new Instances(setU); // O sub conjunto L' tem todos os
										// elementos do conjunto U
		// Renomeando o sub conjunto L'
		subSetL.setRelationName("Sub set L'");
		// Definindo o atributo classe.
		subSetL.setClassIndex(classIndex);

		// Classificando com H
		for (int i = 0; i < subSetL.numInstances(); i++) {
			double classe = classifierH.classifyInstance(subSetL.instance(i));
			String valor = subSetL.instance(i)
					.attribute(subSetL.numAttributes() - 1).value((int) classe);
			subSetL.instance(i).setValue(subSetL.numAttributes() - 1, valor);
		}
	}

	/**
	 * Escolhe as melhores intï¿½ncias dos exemplos que foram rotulados (U')
	 * para serem adicionados ao conjunto de exemplos rotulados (L)
	 * 
	 * @throws Exception
	 * 
	 */
	public void melhoresExemplos(int quantidade) throws Exception {

		if (setU.size() == 0) // se nï¿½o houver exemplos para incorporar...
								// retorne
			return;

		ArrayList<InstanceMeta> listMeta = new ArrayList<InstanceMeta>(); // coleï¿½ï¿½o
																			// de
																			// objetos
																			// com
																			// meta-informaï¿½ï¿½es
																			// sobre
																			// as
																			// instï¿½ncias
		InstanceMeta instanceMeta; // objeto com meta-informaï¿½ï¿½es de cada
									// instï¿½ncia

		// laï¿½o para construir a lista de meta-informaï¿½ï¿½es
		for (int i = 0; i < subSetL.size(); i++) {
			double distClass[] = classifierH.distributionForInstance(subSetL
					.instance(i));

			instanceMeta = new InstanceMeta(); // instanciando o objeto
			instanceMeta.setIntanceIndex(i); // recebe o ï¿½ndice da instï¿½ncia
			instanceMeta.setMaxClassValue(distClass); // recebe a
														// distribuiï¿½ï¿½o
														// pelos grupos da dada
														// instancia

			listMeta.add(instanceMeta); // adiciona o objeto criado ï¿½ lista de
										// meta-dados
		}

		listMeta = quicksort(listMeta); // classifica a lista de meta-dados pelo
										// grau de pertinencia da classe

		int index = 0; // indice da instï¿½ncia
		int hits = quantidade; // quantidade de exemplos que serï¿½o
								// incorporados por vez

		ArrayList<Integer> lista = new ArrayList<Integer>(); // lista que
																// conterï¿½ os
																// indices das
																// instancias
																// para auxiliar
																// na
																// remoï¿½ï¿½o

		while (hits > 0) {
			instanceMeta = listMeta.get(index);

			lista.add(instanceMeta.getIntanceIndex());
			// mapeando a instancia no subSetL a partir de seu ï¿½ndice e
			// adicionando-a no subSetL
			setL.add(subSetL.instance(instanceMeta.getIntanceIndex()));

			index++;
			hits--;
		}

		/*
		 * ordena os ï¿½ndices decrescentemente para que a exclusï¿½o de uma
		 * instancia nï¿½o modifique a ordem das demais instancias que tambï¿½m
		 * serï¿½o excluï¿½das
		 */
		Collections.sort(lista, Collections.reverseOrder());

		for (int i = 0; i < lista.size(); i++) {
			if (setU.size() > lista.get(i))
				setU.delete(lista.get(i)); // apagando do conjunto U as
											// instï¿½ncias que jï¿½ foram
											// incorporadas
		}
	}

	/**
	 * Mï¿½todo para imprimir informaï¿½ï¿½es do classificador e matriz de
	 * confusï¿½o.
	 * 
	 * @param test
	 *            Evaluation - Avaliaï¿½ï¿½o disponivel depois de gerar o
	 *            classificador.
	 */
	private void imprimeResultadoH(Evaluation test) {
		/*
		 * String strSummary="";
		 * 
		 * try { strSummary = test.toSummaryString()+"\n"+test.toMatrixString();
		 * } catch (Exception e) { // TODO Auto-generated catch block
		 * e.printStackTrace(); }
		 * 
		 * 
		 * if ( modoTreino == 1){ try { strSummary = test.toSummaryString("=== "
		 * + folds +
		 * "-fold Cross-validation ===",false)+"\n"+test.toMatrixString(); }
		 * catch (Exception e) { // TODO Auto-generated catch block
		 * e.printStackTrace(); } }
		 * 
		 * m_OutText.setText(m_OutText.getText()+"\n"+strSummary+"\n");
		 */
	}

	public void buildClusterer() throws Exception {
		// CriaÃ§Ã£o do agrupador
		clusterer = algoAgrupamento;
		// atribuindo o algoritmo a classe responsÃ¡vel por agrupar
		clusterer.buildClusterer(setU); // construindo agrupador...
	}

	/**
	 * Aplica tÃ©cnicas de agrupamento para otimizar a classificaÃ§Ã£o dos
	 * exemplos nÃ£o-rotulados. Este mÃ©todo incorpora Ã  base rotulada uma
	 * porÃ§Ã£o de instancias por vez, onde essa quantidade deve fornecida em
	 * percentual.
	 * 
	 * @param percent
	 *            o percentual de exemplos/grupo | o percentual de exemplos do
	 *            conjunto
	 * @param type
	 * @throws Exception
	 */
	public void cluster(double percent, boolean type) throws Exception {

		int numClusters = setL.numClasses(); // armazena a quantidade de grupos
		int[] qtdExemplosPorGrupo = new int[numClusters]; // armazena a
															// quantidade de
															// instancias/grupo
		int countUnlabeledInstances = setU.size(); // armazena a quantidade de
													// instancias nÃ£o-rotuladas

		if (type) { // se o tipo do cluster Ã© 1: retire x elementos de cada
					// grupo
			int qtd = (int) Math.ceil(percent * countUnlabeledInstances
					/ numClusters);
			for (int i = 0; i < numClusters; i++) {
				qtdExemplosPorGrupo[i] = qtd;
			}
		}

		// gera novo classificador com base nos exemplos adicionados ao conjunto
		// de dados rotulados
		criaClassificadorH(algoClassificacao);
		try {
			classificaU();
		} catch (Exception e) {
			e.printStackTrace();
		}

		// faça... enquanto houver elementos a serem incorporados
		while (!subSetL.isEmpty()) {
			// define um grupo para cada instancia do conjunto j� rotulado

			ArrayList<InstanceMeta> completeSet = clusterLabeledInstances();

			// incluindo cada instancia em seu respectivo grupo
			// ordenando os grupos com base no grau de pertinencia da classe dos
			// elementos
			// adicionando o grupo (ordenado) ao conjunto de grupos
			ArrayList<ArrayList> clusters = new ArrayList<ArrayList>(); // coleÃ§Ã£o
																		// de
																		// grupos
			for (int i = 0; i < numClusters; i++) {
				ArrayList<InstanceMeta> cluster; // grupo individual

				cluster = createCluster(completeSet, i); // cria novo grupo a
															// partir do
															// conjunto completo
															// + Ã­ndice do
															// grupo
				clusters.add(quicksort(cluster));

				// se o tipo do cluster Ã© 0: retire x% dos elementos de cada
				// grupo
				// preenche o vetor com a quantidade de exemplos para este grupo
				if (!type && (setU.size() == countUnlabeledInstances))
					qtdExemplosPorGrupo[i] = (int) Math.round(percent
							* cluster.size());
			}

			// teste
			printResults(clusters);

			// incorporando elementos de cada grupo Ã  base rotulada
			for (int i = 0; i < numClusters; i++) {
				if (qtdExemplosPorGrupo[i] > clusters.get(i).size())
					qtdExemplosPorGrupo[i] = clusters.get(i).size();
				incorporate(clusters.get(i), qtdExemplosPorGrupo[i]);
			}

			// gera novo classificador com base nos exemplos adicionados ao
			// conjunto de dados rotulados
			criaClassificadorH(algoClassificacao);
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
	 * Agrupa os exemplos rotulados desconsiderando o valor de seu rÃ³tulo.
	 * 
	 * @return um conjunto de instancias agrupadas
	 * @throws Exception
	 */
	private ArrayList<InstanceMeta> clusterLabeledInstances() throws Exception {
		int clusterValue; // grupo da instancia i
		double classValue; // classe da instancia i
		double distClass[]; // distribuiÃ§Ã£o dos grupos para a da da instancia

		ArrayList<InstanceMeta> instancesRFull = new ArrayList<InstanceMeta>(); // armazena
																				// instanceClass

		// laÃ§o para adicionar cada instancia (com o valor do rÃ³tulo) a um
		// conjunto
		for (int i = 0; i < subSetL.size(); i++) {
			classValue = subSetL.instance(i).classValue();
			subSetL.instance(i).setClassMissing(); // oculta o valor da classe
			clusterValue = clusterer.clusterInstance(subSetL.instance(i)); // agrupa
																			// o
																			// exemplo
			distClass = classifierH
					.distributionForInstance(subSetL.instance(i));
			subSetL.instance(i).setClassValue(classValue); // torna a classe
															// visÃ­vel
															// novamente

			// armazena os valores do Ã­ndice, do grupo, e da distribuiÃ§Ã£o das
			// classes para a dada instancia
			InstanceMeta instanceMeta = new InstanceMeta(i, classValue,
					clusterValue, distClass);

			instancesRFull.add(instanceMeta); // todos os exemplos do conjunto
												// subSetL agrupados
		}

		return instancesRFull;
	}

	private ArrayList<InstanceMeta> createCluster(
			ArrayList<InstanceMeta> instancesRFull, int i) {
		ArrayList<InstanceMeta> cluster = new ArrayList<InstanceMeta>(); // criaÃ§Ã£o
																			// do
																			// grupo

		for (int j = 0; j < instancesRFull.size(); j++) { // laÃ§o para os
															// exemplos
															// rotulados e
															// agrupados
			double clusterValue = instancesRFull.get(j).getClusterValue();

			if (clusterValue == i)
				cluster.add(instancesRFull.get(j));
		}

		return cluster;
	}

	private void incorporate(ArrayList<InstanceMeta> cluster, int numberSamples) {
		InstanceMeta instance;

		for (int i = numberSamples - 1; i >= 0; i--) {
			instance = cluster.get(i);

			int index = instance.getIntanceIndex();
			// System.out.println("Elementos escolhidos: "+index);

			setL.add(subSetL.get(index)); // mapeando a instancia pelo seu
											// indice, e incorporando-a
			setU.delete(i); // removendo elementos do subSetL que jÃ¡ foram
							// incorporados ao setL
			// subSetL.delete(index);
		} // fim do for
	}

	/**
	 * This method sort the input ArrayList using quick sort algorithm.
	 * 
	 * @param clusterInstancesR
	 *            the ArrayList.
	 * @return sorted ArrayList of integers.
	 */
	private ArrayList<InstanceMeta> quicksort(
			ArrayList<InstanceMeta> clusterInstancesR) {

		if (clusterInstancesR.size() <= 1)
			return clusterInstancesR;

		int middle = (int) Math.ceil((double) clusterInstancesR.size() / 2);
		InstanceMeta pivot = clusterInstancesR.get(middle);

		ArrayList<InstanceMeta> less = new ArrayList<InstanceMeta>();
		ArrayList<InstanceMeta> greater = new ArrayList<InstanceMeta>();

		for (int i = 0; i < clusterInstancesR.size(); i++) {
			if (clusterInstancesR.get(i).getMaxClassValue() <= pivot
					.getMaxClassValue()) {
				if (i == middle)
					continue;
				less.add(clusterInstancesR.get(i));
			} else {
				greater.add(clusterInstancesR.get(i));
			}
		}

		return concatenate(quicksort(greater), pivot, quicksort(less));
	}

	/**
	 * Join the less array, pivot integer, and greater array to single array.
	 * 
	 * @param less
	 *            integer ArrayList with values less than pivot.
	 * @param pivot
	 *            the pivot integer.
	 * @param greater
	 *            integer ArrayList with values greater than pivot.
	 * @return the integer ArrayList after join.
	 */
	private ArrayList<InstanceMeta> concatenate(
			ArrayList<InstanceMeta> greater, InstanceMeta pivot,
			ArrayList<InstanceMeta> less) {

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

	/**
	 * MÃ©todo destinado a setar como missing o valor da classe de determinada
	 * porcentagem de instÃ¢ncias do conjunto de dados escolhido pelo usuÃ¡rio.
	 */
	private Instances testeSelection(int porcentagem) {

		// Copiando todos os registros do cojunto original para meu conjunto de
		// testes.
		Instances trainingSetMissing = new Instances(setOriginal);

		// Capturando a representaÃ§Ã£o da quantidade de registros dada a
		// porcentagem.
		int registros = (int) ((trainingSetMissing.numInstances() * porcentagem) / 100);

		/*
		 * O laÃ§o de repetiÃ§Ã£o vai diminuindo a quantidade de registros atÃ©
		 * a porcentagem presente do grupo seja exatamente a que o usuÃ¡rio
		 * determinou
		 */

		// Detalhe importante, registros Ã© o numero total de instÃ¢ncias que
		// deveram ter suas
		// classes setadas como missing.
		// Registro no singular Ã© um index selecionado de maneira aleatÃ³ria
		// para setar no
		// conjunto de dados uma instÃ¢ncia com a classe missing.
		Random random = new Random();
		int registro;

		// limpando m_outText
		m_OutText.setText("");

		if (printFinalTeste) {
			out("\n Selecting a group of instances of the original data set. \n "
					+ "================================================================================");

			out(" Full instances : " + trainingSetMissing.numInstances()
					+ " Full selected instances without label:" + registros);
			out(" Starting random selection...");
		}

		ArrayList<Integer> saiu = new ArrayList<Integer>();
		boolean check;

		while (registros > 0) {

			registro = random.nextInt(trainingSetMissing.numInstances());

			// Certificando que o random sempre escolha alguÃ©m diferente.
			check = false;
			if (saiu.size() > 0) {
				for (int i : saiu) {
					if (i == registro) {
						check = true;
						break;
					}
				}
			}

			if (!check) {
				trainingSetMissing.get(registro).setMissing(classIndex);
				saiu.add(registro);
				registros--;
			}
		}

		if (printFinalTeste) {
			out(" Success! Set of test data was selected successfully. \n "
					+ "end selection \n"
					+ "================================================================================ \n");
		}

		return trainingSetMissing;
	}

	public void separaConjunto2(Instances setOriginal) {

		// Cria o sub-conjunto L
		setL = new Instances(setOriginal); // cria uma c�pia do conjunto
											// original
		setL.setRelationName("Set L"); // nomeia o sub-conjunto L

		// Cria o sub-conjunto U
		setU = new Instances(setOriginal); // cria uma c�pia do conjunto
											// original
		setU.setRelationName("Set U"); // nomeia o sub-conjunto U

		// Percorre todo o sub-conjunto L para eliminar todos os elementos
		// sem-r�tulos
		for (int i = 0; i < setL.numInstances(); i++) {
			// verifica se o valor do atributo classe � null
			if (setL.instance(i).isMissing(setL.numAttributes() - 1)) {
				setL.delete(i);
				i--;
			}
		}

		// Percorre todo o sub-conjunto U para eliminar todos os elementos com
		// r�tulos
		for (int i = 0; i < setU.numInstances(); i++) {
			// verifica se o valor do atributo classe � diferente de null
			if (!setU.instance(i).isMissing(setU.numAttributes() - 1)) {
				setU.delete(i);
				i--;
			}
		}
	}

	/**
	 * MÃ©todo destinado para a realizaÃ§Ã£oo do teste final utilizando o
	 * cojunto gerado a partir do cojunto de dados trainingSetMissing criado a
	 * partir da execuÃ§Ã£o do mÃ©todo testeSelection.
	 */
	private void testeResult() {

		double classe = 0;

		out("\nTesting the results of the classifiers, final test.\n"
				+ "This test compares the label indicated by the classifiers with the actual value of the original set. \n"
				+ "================================================================================\n"
				+ "Creating subset..");
		out("Classifying instances..");

		separaConjunto(setOriginal);

		setOriginal.setClassIndex(classIndex);
		setL.setClassIndex(classIndex);

		// Classificando com o Classificador 1
		for (Instance registro : setL) {

			try {
				classe = classifierH.classifyInstance(registro);
			} catch (Exception e) {
				e.printStackTrace();
				return;
			}

			// Capturando o valor da classe indicada pelo classificador
			String valor = registro.attribute(registro.classIndex()).value(
					(int) classe);
			// Atribuindo a instÃ¢ncia o valor da sua classe
			registro.setValue(registro.classIndex(), valor);
		}

		out("Analyzing the results..");

		int corretamenteClassificadosH = 0;
		for (int i = 0; i < setOriginal.numInstances(); i++) {
			// Verificando se a classe sugerida pelo classificador 1 condiz com
			// o valor verdadeiro da classe.

			if (setOriginal.instance(i).stringValue(setOriginal.classIndex())
					.equals(setL.instance(i).stringValue(setL.classIndex()))) {
				corretamenteClassificadosH++;
			}
		}

		out("Correctly Classified Instances     "
				+ ((100 * corretamenteClassificadosH) / setOriginal
						.numInstances()) + " % \n"
				+ "		Correctly classified of H :" + corretamenteClassificadosH
				+ "\n");

		out("end test \n"
				+ "================================================================================ \n");
	}

	/**
	 * @return the setOriginal
	 */
	public Instances getSetOriginal() {
		return setOriginal;
	}

	/**
	 * @param setOriginal
	 *            ;
	 */
	public void setSetOriginal(Instances setOriginal) {
		this.setOriginal = setOriginal;
	}

	public int getModoTreino() {
		return modoTreino;
	}

	public void setModoTreino(int modoTreino) {
		this.modoTreino = modoTreino;
	}

	public Classifier getAlgoClassificacao() {
		return algoClassificacao;
	}

	public void setAlgoClassificacao(Classifier algoClassificacao) {
		this.algoClassificacao = algoClassificacao;
	}

	public int getClassIndex() {
		return classIndex;
	}

	public void setClassIndex(int classIndex) {
		this.classIndex = classIndex;
	}

	public JTextPane getM_OutText() {
		return m_OutText;
	}

	public void setM_OutText(JTextPane m_OutText) {
		this.m_OutText = m_OutText;
	}

	/**
	 * @return the classifierH
	 */
	public Classifier getClassifierH() {
		return classifierH;
	}

	/**
	 * @param classifierH
	 *            the classifierH to set
	 */
	public void setClassifierH(Classifier classifierH) {
		this.classifierH = classifierH;
	}

	protected static Evaluation setupEval(Evaluation eval,
			Classifier classifier, Instances inst, CostMatrix costMatrix,
			ClassifierErrorsPlotInstances plotInstances,
			AbstractOutput classificationOutput, boolean onlySetPriors)
			throws Exception {

		if (classifier instanceof weka.classifiers.misc.InputMappedClassifier) {
			Instances mappedClassifierHeader = ((weka.classifiers.misc.InputMappedClassifier) classifier)
					.getModelHeader(new Instances(inst, 0));

			if (classificationOutput != null) {
				classificationOutput.setHeader(mappedClassifierHeader);
			}

			if (!onlySetPriors) {
				if (costMatrix != null) {
					eval = new Evaluation(new Instances(mappedClassifierHeader,
							0), costMatrix);
				} else {
					eval = new Evaluation(new Instances(mappedClassifierHeader,
							0));
				}
			}

			if (!eval.getHeader().equalHeaders(inst)) {
				// When the InputMappedClassifier is loading a model,
				// we need to make a new dataset that maps the training
				// instances to
				// the structure expected by the mapped classifier - this is
				// only
				// to ensure that the structure and priors computed by
				// evaluation object is correct with respect to the mapped
				// classifier
				Instances mappedClassifierDataset = ((weka.classifiers.misc.InputMappedClassifier) classifier)
						.getModelHeader(new Instances(mappedClassifierHeader, 0));
				for (int zz = 0; zz < inst.numInstances(); zz++) {
					Instance mapped = ((weka.classifiers.misc.InputMappedClassifier) classifier)
							.constructMappedInstance(inst.instance(zz));
					mappedClassifierDataset.add(mapped);
				}
				eval.setPriors(mappedClassifierDataset);
				if (!onlySetPriors) {
					if (plotInstances != null) {
						plotInstances.setInstances(mappedClassifierDataset);
						plotInstances.setClassifier(classifier);
						/*
						 * int mappedClass =
						 * ((weka.classifiers.misc.InputMappedClassifier
						 * )classifier).getMappedClassIndex();
						 * System.err.println("Mapped class index " +
						 * mappedClass);
						 */
						plotInstances.setClassIndex(mappedClassifierDataset
								.classIndex());
						plotInstances.setEvaluation(eval);
					}
				}
			} else {
				eval.setPriors(inst);
				if (!onlySetPriors) {
					if (plotInstances != null) {
						plotInstances.setInstances(inst);
						plotInstances.setClassifier(classifier);
						plotInstances.setClassIndex(inst.classIndex());
						plotInstances.setEvaluation(eval);
					}
				}
			}
		} else {
			eval.setPriors(inst);
			if (!onlySetPriors) {
				if (plotInstances != null) {
					plotInstances.setInstances(inst);
					plotInstances.setClassifier(classifier);
					plotInstances.setClassIndex(inst.classIndex());
					plotInstances.setEvaluation(eval);
				}
			}
		}

		return eval;
	}

	/**
	 * @return the percent
	 */
	public double getPercent() {
		return percent;
	}

	/**
	 * @param percent
	 *            the percent to set
	 */
	public void setPercent(double percent) {
		this.percent = percent;
	}

	public void out(String text) {
		m_OutText.setText(m_OutText.getText() + text + "\n");
	}

	public void setFolds(int folds) {
		this.folds = folds;
	}

	public void setPrintFinalTeste(boolean printFinalTeste) {
		this.printFinalTeste = printFinalTeste;
	}

	public void setPercentageUnlabeled(int percentageUnlabeled) {
		this.percentageUnlabeled = percentageUnlabeled;
	}

	/**
	 * @return the clusterer
	 */
	public Clusterer getClusterer() {
		return clusterer;
	}

	/**
	 * @param clusterer
	 *            the clusterer to set
	 */
	public void setClusterer(Clusterer clusterer) {
		this.clusterer = clusterer;
	}

	/**
	 * @return the clustererVersion
	 */
	public boolean isClustererVersion() {
		return clustererVersion;
	}

	/**
	 * @param clustererVersion
	 *            the clustererVersion to set
	 */
	public void setClustererVersion(boolean clustererVersion) {
		this.clustererVersion = clustererVersion;
	}

	/**
	 * @return the algoAgrupamento
	 */
	public Clusterer getAlgoAgrupamento() {
		return algoAgrupamento;
	}

	/**
	 * @param algoAgrupamento
	 *            the algoAgrupamento to set
	 */
	public void setAlgoAgrupamento(Clusterer algoAgrupamento) {
		this.algoAgrupamento = algoAgrupamento;
	}

	/**
	 * @return the incorporationMode
	 */
	public boolean isIncorporationMode() {
		return incorporationMode;
	}

	/**
	 * @param incorporationMode
	 *            the incorporationMode to set
	 */
	public void setIncorporationMode(boolean incorporationMode) {
		this.incorporationMode = incorporationMode;
	}

	/**
	 * @return the folds
	 */
	public int getFolds() {
		return folds;
	}

	/**
	 * @return the percentageUnlabeled
	 */
	public int getPercentageUnlabeled() {
		return percentageUnlabeled;
	}

	public void setLog(Logger log) {
		m_Log = log;
	}

	public Evaluation getETest() {
		return this.eTest;
	}

	public void setETest(Evaluation eval) {
		this.eTest = eval;
	}

	public double getPctCorreto() {
		setPctCorreto(eTest.pctCorrect());
		return this.pctCorreto;
	}

	public void setPctCorreto(double pctCorreto) {
		this.pctCorreto = pctCorreto;
	}

}