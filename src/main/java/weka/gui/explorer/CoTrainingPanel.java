package weka.gui.explorer;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.GridLayout;
import java.awt.event.InputEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.beans.PropertyChangeSupport;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Random;
import java.util.Vector;

import javax.swing.ButtonGroup;
import javax.swing.DefaultComboBoxModel;
import javax.swing.GroupLayout;
import javax.swing.GroupLayout.Alignment;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JMenuItem;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JPopupMenu;
import javax.swing.JRadioButton;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.JTextField;
import javax.swing.JTextPane;
import javax.swing.UIManager;
import javax.swing.border.TitledBorder;
import javax.swing.table.DefaultTableModel;

import weka.associations.Associator;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.CostMatrix;
import weka.classifiers.Evaluation;
import weka.classifiers.Sourcable;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.output.prediction.AbstractOutput;
import weka.classifiers.evaluation.output.prediction.Null;
import weka.clusterers.Clusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.CapabilitiesHandler;
import weka.core.Drawable;
import weka.core.Environment;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.SerializedObject;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.Loader;
import weka.gui.CostMatrixEditor;
import weka.gui.GenericObjectEditor;
import weka.gui.Logger;
import weka.gui.PropertyPanel;
import weka.gui.ResultHistoryPanel;
import weka.gui.SysErrLog;
import weka.gui.TaskLogger;
import weka.gui.explorer.Explorer.ExplorerPanel;
import weka.gui.explorer.Explorer.LogHandler;
import weka.learning.semisupervised.ClusterSet;
import weka.learning.semisupervised.CoTraining;
import weka.learning.semisupervised.SelfTraining;

public class CoTrainingPanel extends JPanel  implements ExplorerPanel, LogHandler{

	 /* Essa variável mantém a conexão com o Explorer do weka */
	 protected Explorer m_Explorer = null;
	 
	 /* Conjunto de dados carregado originalmente pelo usuário */
	 private Instances m_Instances;
	 private CoTraining coTraining;
	 
	 /* Panel de texto(textPane) onde o resultado é mostrado para u usuário*/
	 private JTextPane m_OutText = new JTextPane();	 
	 
	 /* Jpanel do lado esquerdo onde fica os outros componentes com as configurações do classificador 1 e 2 */
	 private JPanel panelLeft;
	
	 /* Jpanel do classificador 1*/
	 private  JPanel panelClass1;
	 
	 /* Atributos para configuração do treinamento dos classificadores*/
	 private JRadioButton m_TrainBut;
	 private JRadioButton m_CVBut;		
	 private JTextField textField_Folds;	 	 
	 private JTextField seedField;
	 private JTextField porcetageUnlabeledField;
	 private JCheckBox  showTesteResult; 
	 private JTextField percetageMEField; // Porcentagem que é adicionada a cada ciclo nos melhores exemplos
	 private JCheckBox useCluster;
	 
	 /* tabela de seleção de atributos do classificador 1*/
	 private JTable tableClass1;
	 
	 /* Modelo que a tabela 1 vai carregar.*/
	 private MyTableModel modelClass1 = new MyTableModel();
	 
	 /* Jpanel do classificador 2*/
	 private JPanel panelClass2;
	 
	 /* tabela de seleção de atributos do classificador 2*/
	 private JTable tableClass2;
	 /* Modelo que a tabela 2 vai carregar.*/
	 private MyTableModel modelClass2 = new MyTableModel();
	
	 
	 /* Combo box respons�vel pela escolha do atributo class do conjunto de dados*/
	 JComboBox comboBox;
	 
	  /* Lets the user configure the classifier. */
	  protected GenericObjectEditor m_ClassifierEditor1 = new GenericObjectEditor();

	  /* The panel showing the current classifier selection. */
	  protected PropertyPanel m_CEPanel = new PropertyPanel(m_ClassifierEditor1);
	 
	  /* Lets the user configure the classifier. */
	  protected GenericObjectEditor m_ClassifierEditor2 = new GenericObjectEditor();

	  /* The panel showing the current classifier selection. */
	  protected PropertyPanel m_CEPanel2 = new PropertyPanel(m_ClassifierEditor2);
	 
	  protected ResultHistoryPanel m_History = new ResultHistoryPanel(m_OutText);
	  
	  
	  /* Click to start running the classifier. */
	  protected JButton m_StartBut = new JButton("Start");

	  /* Click to stop a running classifier. */
	  protected JButton m_StopBut = new JButton("Stop");
	  
	  /* The destination for log/status messages. */
	  protected Logger m_Log ;	  
	  
	  private Thread exe = null;
	  	 	  	 
	  private GenericObjectEditor m_ClusterEditor = new GenericObjectEditor();
	  protected PropertyPanel m_PanelAssociator = new PropertyPanel(m_ClusterEditor);
	  	  
	  	  
	  
	  /**
	  * @wbp.parser.entryPoint
	  */
	 public CoTrainingPanel(){
		    		 		 
		    setLayout(new GridLayout()); 
		    
		    JPanel panelP = new JPanel();			   
		    add(panelP);		   
		    
		    System.out.println("draw");
		    
		    /*
		     * 	CROLLPANE QUE CONTÉM A SAÍDA DE TEXTO DA EXECUÇÃO 
		     */
		    JScrollPane scrollPane = new JScrollPane();
		    scrollPane.setBorder(new TitledBorder(null, "Co-training output", TitledBorder.LEADING, TitledBorder.TOP, null, null));
		    scrollPane.setViewportView(m_OutText);
		    
		    
		    /*
		     *  SCROLLPANE RESPONSÁVEL POR GUARDAR O PANELLEFT
		     */
		    JScrollPane scrollPane_2 = new JScrollPane();
		    scrollPane_2.setBorder(null);
		    
		    // PanelLeft, panel que contém os panels dos classficadores
		    panelLeft = new JPanel();		    
		    scrollPane_2.setViewportView(panelLeft);
		    panelClass1 = new JPanel();
		    panelClass1.setBorder(new TitledBorder(UIManager.getBorder("TitledBorder.border"), "Classifier 1", TitledBorder.LEADING, TitledBorder.TOP, null, null));
		    
		    m_ClassifierEditor1.setClassType(Classifier.class);
		    m_ClassifierEditor1.setValue(ExplorerDefaults.getClassifier());
		    m_ClassifierEditor1.addPropertyChangeListener(new PropertyChangeListener() {
		      
		      public void propertyChange(PropertyChangeEvent e) {
		       
		        // Check capabilities        
		        Capabilities currentFilter = m_ClassifierEditor1.getCapabilitiesFilter();
		        Classifier classifier = (Classifier) m_ClassifierEditor1.getValue();
		        Capabilities currentSchemeCapabilities = null;
		        if (classifier != null && currentFilter != null
		            && (classifier instanceof CapabilitiesHandler)) {
		          currentSchemeCapabilities = ((CapabilitiesHandler) classifier)
		              .getCapabilities();

		          if (!currentSchemeCapabilities.supportsMaybe(currentFilter)
		              && !currentSchemeCapabilities.supports(currentFilter)) {
		            
		          }
		        }
		        repaint();
		      }
		    });
		    
		    
		    m_ClassifierEditor2.setClassType(Classifier.class);
		    m_ClassifierEditor2.setValue(ExplorerDefaults.getClassifier());
		    m_ClassifierEditor2.addPropertyChangeListener(new PropertyChangeListener() {
		      
		      public void propertyChange(PropertyChangeEvent e) {
		       
		        // Check capabilities        
		        Capabilities currentFilter = m_ClassifierEditor2.getCapabilitiesFilter();
		        Classifier classifier = (Classifier) m_ClassifierEditor2.getValue();
		        Capabilities currentSchemeCapabilities = null;
		        if (classifier != null && currentFilter != null
		            && (classifier instanceof CapabilitiesHandler)) {
		          currentSchemeCapabilities = ((CapabilitiesHandler) classifier)
		              .getCapabilities();

		          if (!currentSchemeCapabilities.supportsMaybe(currentFilter)
		              && !currentSchemeCapabilities.supports(currentFilter)) {
		            
		          }
		        }
		        repaint();
		      }
		    });
		    
		    /*
		     * Panel responsável por mostrar opções que podem ser alteradas pelo usuário para a execução do algoritmo co-training.
		     */
		    JPanel optionsPanel = new JPanel();
		    optionsPanel.setBorder(new TitledBorder(UIManager.getBorder("TitledBorder.border"),
		    		"training options", TitledBorder.LEADING, TitledBorder.TOP, null, null));		    
		    
		    ButtonGroup bg = new ButtonGroup();  
		    
		    m_TrainBut = new JRadioButton("Use traning set");
		    m_TrainBut.setBounds(20, 20, 150, 23);		    
		    bg.add(m_TrainBut);		   
		    
		    
		    m_CVBut = new JRadioButton("Cross- validation     Folds");
		    m_CVBut.setBounds(20, 40, 210, 23);
		    m_CVBut.setSelected(true);
		    bg.add(m_CVBut);
		    		    
		    textField_Folds = new JTextField("10");
		    textField_Folds.setBounds(230, 40, 40, 20);		
		    
		    JLabel seedLabel = new JLabel("Seed");
		    seedLabel.setBounds(40, 60, 50, 23);
		    
		    seedField = new JTextField("30");
		    seedField.setBounds(95,65, 40, 20);
		    
		    JLabel settingsTest = new JLabel("Settings for selection of instances");
		    settingsTest.setBounds(20, 90, 250, 23);
		    settingsTest.setForeground(Color.gray);
		    
		    JLabel porcetageUnlabeledLabel = new JLabel("Percentage without labels %");
		    porcetageUnlabeledLabel.setBounds(20, 109, 230, 23);
		    porcetageUnlabeledField = new JTextField("10");
		    porcetageUnlabeledField.setBounds(230,109, 40, 20);
		    
		    showTesteResult =  new JCheckBox("Comparative test results showing");
		    showTesteResult.setBounds(20,130,280,20);
		    showTesteResult.setSelected(true);
		    
		    JLabel incrementLabel = new JLabel("Increment per cycle");
		    incrementLabel.setBounds(20, 155, 230, 23);
		    incrementLabel.setForeground(Color.gray);
		    
		    JLabel percetageMELabel = new JLabel("Best examples %");
		    percetageMELabel.setBounds(20,175,150,20);
		    percetageMEField  = new JTextField("10");
		    percetageMEField.setBounds(150,175, 40, 20);
		    
		    
		    useCluster =  new JCheckBox("Use clusters to separate attributes");
		    useCluster.setBounds(20,210,280,20);
		    useCluster.setSelected(true);
		    
		    m_PanelAssociator.setBounds(10,230,300, 25);
		    m_ClusterEditor.setClassType(Clusterer.class);
		    m_ClusterEditor.setValue(new SimpleKMeans());//ExplorerDefaults.getClusterer()
		    m_ClusterEditor.addPropertyChangeListener(new PropertyChangeListener() {
		      public void propertyChange(PropertyChangeEvent e) {
		        m_StartBut.setEnabled(true);
		        Capabilities currentFilter = m_ClusterEditor.getCapabilitiesFilter();
		        Clusterer clusterer = (Clusterer) m_ClusterEditor.getValue();
		        Capabilities currentSchemeCapabilities =  null;
		        if (clusterer != null && currentFilter != null && 
		            (clusterer instanceof CapabilitiesHandler)) {
		          currentSchemeCapabilities = ((CapabilitiesHandler)clusterer).getCapabilities();
		          
		          if (!currentSchemeCapabilities.supportsMaybe(currentFilter) &&
		              !currentSchemeCapabilities.supports(currentFilter)) {
		            m_StartBut.setEnabled(false);
		          }
		        }
			repaint();
		      }
		    });
		    		    		    
		    optionsPanel.setLayout(null);
		    optionsPanel.add(m_TrainBut);		   
		    optionsPanel.add(m_CVBut);		   		    
		    optionsPanel.add(textField_Folds);	
		    optionsPanel.add(seedLabel);
		    optionsPanel.add(seedField);
		    optionsPanel.add(settingsTest);
		    optionsPanel.add(porcetageUnlabeledLabel);
		    optionsPanel.add(porcetageUnlabeledField);
		    optionsPanel.add(showTesteResult);
		    optionsPanel.add(incrementLabel);  
		    optionsPanel.add(percetageMELabel); 
		    optionsPanel.add(percetageMEField); 
		    optionsPanel.add(useCluster);
		    optionsPanel.add(m_PanelAssociator);
		    
		    /*
		     * ScrollPane e Tabela de seleção de atributos do classificador 1
		     */
		    JScrollPane scrollPaneClass1 = new JScrollPane();
		    scrollPaneClass1.setBounds(10, 55, 300, 160);
		    scrollPaneClass1.setBorder(new TitledBorder(null, "Select atributes", TitledBorder.LEADING, TitledBorder.TOP, null, null));				    		    		       		      
		    tableClass1 = new JTable(modelClass1);
		    scrollPaneClass1.setViewportView(tableClass1);
		    
		    
		    /*
		     * CRIANDO O PANEL DO CLASSIFICADOR 2
		     */
		    panelClass2 = new JPanel();
		    panelClass2.setBorder(new TitledBorder(UIManager.getBorder("TitledBorder.border"), "Classifier 2", TitledBorder.LEADING, TitledBorder.TOP, null, null));		    		    		 		   
		    
		    /*
		     * ScrollPane e Tabela de seleção de atributos do classificador 2
		     */
		    JScrollPane scrollPane2 = new JScrollPane();
		    tableClass2 = new JTable(modelClass2);
		    scrollPane2.setViewportView(tableClass2);
		    scrollPane2.setBounds(10, 55, 300, 160);
		    scrollPane2.setBorder(new TitledBorder(null, "Select atributes", TitledBorder.LEADING, TitledBorder.TOP, null, null));
		  
		    
		
		    JPanel panelComfirm = new JPanel();
		    panelComfirm.setBorder(null);
		    panelComfirm.setBounds(72, 57, 319, 210);
			add(panelComfirm);
			panelComfirm.setLayout(null);
			
			comboBox = new JComboBox();
			comboBox.setBounds(0, 0, 319, 34);
			panelComfirm.add(comboBox);
			
			m_StartBut = new JButton("Start");
			m_StartBut.addMouseListener(new MouseAdapter() {
				
				public void mouseReleased(MouseEvent arg0) {
					
					//obtendo algoritimos classiificadores da �rvore					
					Classifier classificador1 = (Classifier) m_ClassifierEditor1.getValue();
					Classifier classificador2 = (Classifier) m_ClassifierEditor2.getValue();
					Instances instances1 = new Instances(m_Instances);
					Instances instances2 = new Instances(m_Instances);
					String saida = "";
					
					/*
					 *  Realizando a sele��o dos atributos que o us�ario deseja que seus dois subconjuntos de L tenham. 
					 *  A partir dessa sele��o � criado duas listas de booleanos que ser� passado como par�metro para o 
					 *  m�todo coTraining. O m�todo a partir dessa lista � capaz de fazer a separa��o desses atributos
					 *  no conjunto de L.
					 */
					
					
					/* Ao inv�s de passar os models da tabela como par�metros, para que o m�todo coTraining n�o
					* ficasse t�o sujeit�vel a inteface foi resolvido passar listas do tipo booleano.
					*/
					ArrayList<Boolean> listaAtributosUm = new ArrayList<Boolean>();
					ArrayList<Boolean> listaAtributosDois = new ArrayList<Boolean>();
																							
					
					/* La�o de repeti��o para realizar a captura dos valores booleanos dos models das tabelas			
					 * tableClass1 � o model da tabela onde o us�ario realizou a sele��o dos atributos.					
					*/								
					for (int i = 0; i < modelClass1.getRowCount() ; i++) {
						
						listaAtributosUm.add((Boolean) modelClass1.getValueAt(i, 1));					
					}
					
					//tableClass2 � o outro model da tabela  2 onde o us�ario realizou a sele��o dos atributos.					
					for (int i = 0; i < modelClass2.getRowCount(); i++) {
						
						listaAtributosDois.add((Boolean) modelClass2.getValueAt(i, 1));						
					}
												
					/*
					 * Validando entradas do usuário
					 */
					int porcentageUnlabeled = Integer.parseInt(porcetageUnlabeledField.getText());
					int folds = Integer.parseInt(textField_Folds.getText());
					int seed = Integer.parseInt(seedField.getText());
					int modTraining = 0;
					int percetageME = Integer.parseInt(percetageMEField.getText());	
					Clusterer cluster = (Clusterer) m_ClusterEditor.getValue();	
					
				if (seed>0 && folds>0 && porcentageUnlabeled>0 && percetageME >0 && percetageME<=100  ){											
					
					if (m_CVBut.isSelected()){
					      modTraining = 1;
					}
						
				// Caso useCluster true , irá separar os atributos usando um algoritmo de agrupamento
				if (useCluster.isSelected() ){
					
					ClusterSet c = new ClusterSet();
					c.SetClusterer(cluster);
					c.buildClusterer(m_Instances,comboBox.getSelectedIndex());
					
					listaAtributosUm = c.getListaAtributosUm();
					listaAtributosDois = c.getListaAtributosDois();
				}
					
					
				   /** Instânciando um objeto do tipo CoTraining.
				    *  É possivél fazer isso de duas maneiras:
				    *  Passando os parâmetros pelo construtor.
				    *  Setando os valores que nem o código abaixo. Dessa maneira é possivel organizar melhor essa etapa para
				    *  não se perder.
				    * */
					coTraining = new CoTraining();					
					coTraining.setSetOriginal(m_Instances); // Setando o conjunto original
					coTraining.setAlgoClassificacao1(classificador1); // Setando o classificador 1 escolhido pelo usuário
					coTraining.setAlgoClassificacao2(classificador2); // Setando o classificador 2  escolhido pelo usuário
					coTraining.setClassIndex(comboBox.getSelectedIndex()); // Setando o atributo alvo escolhido pelo usuário
					coTraining.setM_OutText(m_OutText); //Setando o JTextPane, o CoTraining enviar suas menssagens de saída para ele. 
					coTraining.setListaAtributosUm(listaAtributosUm); // Setando a lista booleana dos atributos escolhidos pelo usuário
					coTraining.setListaAtributosDois(listaAtributosDois); // Setando a lista booleana dos atributos escolhidos pelo usuário
					coTraining.setPrintFinalTeste(showTesteResult.isSelected()); // Caso seja verdadeiro irá mostrar o resultado do teste final de comparação dos conjuntos
					coTraining.setPercentageUnlabeled(porcentageUnlabeled); // Definindo a porcetegem de intâncias que teram seus rótulos retirados para o treinamento.
					coTraining.setFolds(folds); // Definindo o numero de folds para o cross- validation
					coTraining.setSeed(seed); // Definindo a semente que será utilizada para o random do cross- validation
					coTraining.setModTraining(modTraining); // Definindo o modo da avaliação do teste dos classificadores;
					coTraining.setPercentMe(percetageME); // Definindo a porcentagem de icremento dos melhores exemplos
					coTraining.setM_Log(m_Log); // Defininfo o objeto Logger 					
					
					if (exe != null ){
						exe.interrupt();
					}
														
					exe= new Thread(coTraining);
					exe.start();
					
				  }else{
					  final JPanel panel = new JPanel();
					 JOptionPane.showMessageDialog(panel, "Mistake of configuration, review the training parameters.", "Error", JOptionPane.ERROR_MESSAGE);
				  }
					
				}
			});
			
			m_StartBut.setBounds(10, 35, 152, 27);
			panelComfirm.add(m_StartBut);
			
			m_StopBut = new JButton("Stop");
			m_StopBut.setBounds(164, 35, 145, 27);
			panelComfirm.add(m_StopBut);
			
			
			m_StopBut.addMouseListener(new MouseAdapter() {			
				public void mouseReleased(MouseEvent arg0) {
					coTraining.kill();
				}
			});
			
			JScrollPane scrollPaneResult = new JScrollPane();
			scrollPaneResult.setBounds(0, 73, 319, 137);
			panelComfirm.add(scrollPaneResult);
			scrollPaneResult.setBorder(new TitledBorder(null, "Result list (rigth click for options)", TitledBorder.LEADING, TitledBorder.TOP, null, null));
						
			scrollPaneResult.setViewportView(m_History);
	    
		    GroupLayout gl_panelLeft = new GroupLayout(panelLeft);
		    gl_panelLeft.setHorizontalGroup(
		    	gl_panelLeft.createParallelGroup(Alignment.LEADING)
		    	    .addComponent(optionsPanel, GroupLayout.PREFERRED_SIZE, 319, GroupLayout.PREFERRED_SIZE)
		    		.addComponent(panelClass1, GroupLayout.PREFERRED_SIZE, 319, GroupLayout.PREFERRED_SIZE)
		    		.addComponent(panelClass2, GroupLayout.PREFERRED_SIZE, 319, GroupLayout.PREFERRED_SIZE)
		    		.addComponent(panelComfirm, GroupLayout.PREFERRED_SIZE, 319, GroupLayout.PREFERRED_SIZE)
		    );
		    gl_panelLeft.setVerticalGroup(
		    	gl_panelLeft.createParallelGroup(Alignment.LEADING)
		    		.addGroup(gl_panelLeft.createSequentialGroup()
		    			.addComponent(optionsPanel, GroupLayout.PREFERRED_SIZE, 270, GroupLayout.PREFERRED_SIZE)
			    		.addGap(6)
		    			.addComponent(panelClass1, GroupLayout.PREFERRED_SIZE, 225, GroupLayout.PREFERRED_SIZE)
		    			.addGap(6)
		    			.addComponent(panelClass2, GroupLayout.PREFERRED_SIZE, 225, GroupLayout.PREFERRED_SIZE)
		    			.addGap(6)
		    			.addComponent(panelComfirm, GroupLayout.PREFERRED_SIZE, 210, GroupLayout.PREFERRED_SIZE))
		    );
  
	 
	   /*
	    * ADICIONANDO OS COMPONENTES AS DUAS PANEL : CLASSIFICADOR 1 E 2
	    */		   
		    panelClass2.setLayout(null);		    
		    panelClass2.add(scrollPane2);
		    
		    panelClass1.setLayout(null);		    		    
		    panelClass1.add(scrollPaneClass1);
		    
		    
		    m_History.setHandleRightClicks(false);
		    // see if we can popup a menu for the selected result
		    m_History.getList().addMouseListener(new MouseAdapter() {
		      
		      public void mouseClicked(MouseEvent e) {
		        if (((e.getModifiers() & InputEvent.BUTTON1_MASK) != InputEvent.BUTTON1_MASK)
		            || e.isAltDown()) {
		          int index = m_History.getList().locationToIndex(e.getPoint());
		          if (index != -1) {
		            String name = m_History.getNameAtIndex(index);
		            
		          } else {
		            
		          }
		        }
		      }
		    });
		    
		    JPanel panel = new JPanel();
		    panel.setBounds(10, 21, 299, 29);
		    panel.setLayout(null);
		    m_CEPanel.setBounds(10, 5, 289, 23);
		    panel.add(m_CEPanel);
		    
		    JPanel panelChoose2 = new JPanel();
		    panelChoose2.setBounds(10, 21, 299, 29);
		    panelChoose2.setLayout(null);
		    m_CEPanel2.setBounds(10, 5, 289, 23);
		    panelChoose2.add(m_CEPanel2);
		    
		    panelClass1.add(panel);
		    panelClass2.add(panelChoose2);
		    panelLeft.setLayout(gl_panelLeft);
		    JButton btnNewButton = new JButton("Choose");		    		    		   		   		    
		    		    
		    btnNewButton.addMouseListener(new MouseAdapter() {
		    	
		    	public void mouseReleased(MouseEvent arg0) {
		    		//implementar aqui a escolhar do algoritmo self-traning ou traning. no entanto a interface n�o est� se adaptando a essa escolha.		
		    	    
		    	}
		    });
		    GroupLayout gl_panelP = new GroupLayout(panelP);
		    gl_panelP.setHorizontalGroup(
		    	gl_panelP.createParallelGroup(Alignment.LEADING)		    		
		    		.addGroup(gl_panelP.createSequentialGroup()
		    			.addComponent(scrollPane_2, GroupLayout.PREFERRED_SIZE, 336, GroupLayout.PREFERRED_SIZE)
		    			.addGap(6)
		    			.addComponent(scrollPane, GroupLayout.DEFAULT_SIZE, 305, Short.MAX_VALUE))
		    );
		    gl_panelP.setVerticalGroup(
		    	gl_panelP.createParallelGroup(Alignment.LEADING)
		    		.addGroup(gl_panelP.createSequentialGroup()		    			
		    			.addGap(6)
		    			.addGroup(gl_panelP.createParallelGroup(Alignment.LEADING)
		    				.addGroup(gl_panelP.createSequentialGroup()
		    					.addComponent(scrollPane_2, GroupLayout.DEFAULT_SIZE, 493, Short.MAX_VALUE)
		    					.addGap(11))
		    				.addComponent(scrollPane, GroupLayout.DEFAULT_SIZE, 504, Short.MAX_VALUE)))
		    );
		    panelP.setLayout(gl_panelP);
	
	 }
	 
	 
	
	public void setExplorer(Explorer parent) {		
		
		 m_Explorer = parent;
	}

	
	public Explorer getExplorer() {
		 return m_Explorer;
	}

	
	public void setInstances(Instances inst) {
		
		//carrega o conjunto de dados escolhido pelo us�ario na primeira aba do weka
	    m_Instances = inst;
	    
	    //Adicionando os atributos do conjunto de dados nas tabelas de sele��o de atributos.
       for( int i= 0 ;i<m_Instances.numAttributes();i++){
    	  
    	   modelClass1.addRow(new Object[]{i+1, true, m_Instances.attribute(i).name()});
    	   modelClass2.addRow(new Object[]{i+1, true, m_Instances.attribute(i).name()});
       }
       
       //Adicionando os atributos no combobox, para permitir a escolha do atributo classe pelo usu�rio
       String[] attribNames = new String[m_Instances.numAttributes()];
       for (int i = 0; i < attribNames.length; i++) {
         String type = "(" + Attribute.typeToStringShort(m_Instances.attribute(i)) + ")";
         attribNames[i] = type + m_Instances.attribute(i).name();
       }
       
       comboBox.setModel(new DefaultComboBoxModel(attribNames));
       
       //dizendo pro comboBox que por padr�o o atributo final ser� o atributo class
       if (attribNames.length > 0) {
         if (inst.classIndex() == -1)
        	 comboBox.setSelectedIndex(attribNames.length - 1);
         else
        	 comboBox.setSelectedIndex(inst.classIndex());
         comboBox.setEnabled(true);
       }
        
		
	}

	
	public String getTabTitle() {
		return "Co-training";
	}

	
	public String getTabTitleToolTip() {
		return "Co-training learning";
	}	 
	  
		 
public class MyTableModel extends DefaultTableModel {

    

	public MyTableModel() {
      super(new String[]{"Num.", " ", "Atribute"}, 0);
    }

    
    public Class<?> getColumnClass(int columnIndex) {
      Class clazz = String.class;
      switch (columnIndex) {
        case 0:
          clazz = Integer.class;
          break;
        case 1:
          clazz = Boolean.class;
          break;
      }
      return clazz;
    }

    
    public boolean isCellEditable(int row, int column) {
      return column == 1;
    }

    
    public void setValueAt(Object aValue, int row, int column) {
      if (aValue instanceof Boolean && column == 1) {        
        Vector rowData = (Vector)getDataVector().get(row);
        rowData.set(1, (Boolean)aValue);
        fireTableCellUpdated(row, column);
      }
    }

  }


public void setLog(Logger newLog) {
	m_Log = newLog;
	
}

public static void main(String [] args) {

    try {
      final javax.swing.JFrame jf =
	new javax.swing.JFrame("Weka Explorer: Classifier");
      jf.getContentPane().setLayout(new BorderLayout());
      final CoTrainingPanel sp = new CoTrainingPanel();
      jf.getContentPane().add(sp, BorderLayout.CENTER);
      weka.gui.LogPanel lp = new weka.gui.LogPanel();
      sp.setLog(lp);
      jf.getContentPane().add(lp, BorderLayout.SOUTH);
      jf.addWindowListener(new java.awt.event.WindowAdapter() {
	public void windowClosing(java.awt.event.WindowEvent e) {
	  jf.dispose();
	  System.exit(0);
	}
      });
      jf.pack();
      jf.setSize(800, 600);
      jf.setVisible(true);
      if (args.length == 1) {
	System.err.println("Loading instances from " + args[0]);
	java.io.Reader r = new java.io.BufferedReader(
			   new java.io.FileReader(args[0]));
	Instances i = new Instances(r);
	sp.setInstances(i);
      }
    } catch (Exception ex) {
      ex.printStackTrace();
      System.err.println(ex.getMessage());
    }
  }
}



