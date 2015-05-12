package weka.gui.explorer;

import java.awt.Color;
import java.awt.GridLayout;
import java.awt.event.InputEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.beans.PropertyChangeSupport;
import java.text.SimpleDateFormat;
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

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.CostMatrix;
import weka.classifiers.Evaluation;
import weka.classifiers.Sourcable;
import weka.classifiers.evaluation.output.prediction.AbstractOutput;
import weka.classifiers.evaluation.output.prediction.Null;
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

public class SemiSupervisedPanel extends JPanel  implements ExplorerPanel, LogHandler{

	 /** Essa vari�vel mant�m a conex�o com o Explorer do weka */
	 protected Explorer m_Explorer = null;
	 
	 /** Conjunto de dados carregado originalmente pelo us�ario */
	 private Instances m_Instances;		
	 
	 /** Panel de texto(textPane) onde o resultado � mostrado para u us�ario*/
	 private JTextPane m_OutText = new JTextPane();	 
	 	
	 /** Jpanel do lado esquerdo onde fica os outros componentes como as configura��es do classificador 1 e 2 */
	 private JPanel panelLeft;
	
	 /** Jpanel do classificador 1*/
	 private  JPanel panelClass1;
	 
	 /** bot�es radio para sele��o do modo como vai ser realizado o treinamento do classificador 1 */
	 private JRadioButton m_CVBut;
	 private JRadioButton m_PercentBut;
	 private JRadioButton m_TrainBut;
	 private JRadioButton m_TestSplitBut;
	 
	 /** bot�es radio para sele��o do modo como vai ser realizado o treinamento do classificador 1 */
	 private JRadioButton m_CVBut2;
	 private JRadioButton m_PercentBut2;
	 private JRadioButton m_TrainBut2;
	 private JRadioButton m_TestSplitBut2;
	 
	 private JTextField textField;
	 
	 // numeros de folds do classificador 1
	 private JTextField textField_Folds;
	 
	 // split porcetagem do classificador 1
	 private JTextField textField_3;
	 
	 /** tabela de sele��o de atributos do classificador 1*/
	 private JTable tableClass1;
	 
	 /** Modelo que a tabela 1 vai carregar.*/
	 private MyTableModel modelClass1 = new MyTableModel();
	 
	 /** Jpanel do classificador 2*/
	 private JPanel panelClass2;
	 
	 /** tabela de sele��o de atributos do classificador 2*/
	 private JTable tableClass2;
	 /** Modelo que a tabela 2 vai carregar.*/
	 private MyTableModel modelClass2 = new MyTableModel();
	 
	// numeros de folds do classificador 2
	 private JTextField textField_Folds2;
	 
	// split porcetagem do classificador 2
	 private JTextField textField_5;
	 
	 /** Combo box respons�vel pela escolha do atributo class do conjunto de dados*/
	 JComboBox comboBox;
	 
	  /** Lets the user configure the classifier. */
	  protected GenericObjectEditor m_ClassifierEditor = new GenericObjectEditor();

	  /** The panel showing the current classifier selection. */
	  protected PropertyPanel m_CEPanel = new PropertyPanel(m_ClassifierEditor);
	 
	  /** Lets the user configure the classifier. */
	  protected GenericObjectEditor m_ClassifierEditor2 = new GenericObjectEditor();

	  /** The panel showing the current classifier selection. */
	  protected PropertyPanel m_CEPanel2 = new PropertyPanel(m_ClassifierEditor);
	 
	  protected ResultHistoryPanel m_History = new ResultHistoryPanel(m_OutText);
	  
	  
	  /** Click to start running the classifier. */
	  protected JButton m_StartBut = new JButton("Start");

	  /** Click to stop a running classifier. */
	  protected JButton m_StopBut = new JButton("Stop");
	  
	  /** The destination for log/status messages. */
	  protected Logger m_Log = new SysErrLog();	  
	  	 	  	 
	  
	  /**
	  * @wbp.parser.entryPoint
	  */
	 public SemiSupervisedPanel(){
		    
		    
		    setLayout(new GridLayout()); 
		    
		    JPanel panelP = new JPanel();			   
		    add(panelP);
		    
		    
		    System.out.println("draw");
		    
		    JScrollPane scrollPane = new JScrollPane();
		    scrollPane.setBorder(new TitledBorder(null, "Semi-supervised output", TitledBorder.LEADING, TitledBorder.TOP, null, null));
		    scrollPane.setViewportView(m_OutText);
		    
		    
		    JPanel panel_3 = new JPanel();
		    panel_3.setBorder(new TitledBorder(null, "Algorithims of learning semi-supervised ", TitledBorder.LEADING, TitledBorder.TOP, null, null));
		    
		    JScrollPane scrollPane_2 = new JScrollPane();
		    scrollPane_2.setBorder(null);
		    
		    
		    
		    panelLeft = new JPanel();
		    
		    scrollPane_2.setViewportView(panelLeft);
		    panelClass1 = new JPanel();
		    panelClass1.setBorder(new TitledBorder(UIManager.getBorder("TitledBorder.border"), "Classifier 1", TitledBorder.LEADING, TitledBorder.TOP, null, null));
		    
		    m_ClassifierEditor.setClassType(Classifier.class);
		    m_ClassifierEditor.setValue(ExplorerDefaults.getClassifier());
		    m_ClassifierEditor.addPropertyChangeListener(new PropertyChangeListener() {
		      
		      public void propertyChange(PropertyChangeEvent e) {
		       
		        // Check capabilities        
		        Capabilities currentFilter = m_ClassifierEditor.getCapabilitiesFilter();
		        Classifier classifier = (Classifier) m_ClassifierEditor.getValue();
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
		    
		    ButtonGroup bg = new ButtonGroup();  
		    
		    m_TrainBut = new JRadioButton("Use traning set");
		    m_TrainBut.setBounds(20, 57, 109, 23);
		    m_TrainBut.setSelected(true);
		    bg.add(m_TrainBut);
		    
		    m_TestSplitBut = new JRadioButton("Supplied test set");
		    m_TestSplitBut.setBounds(20, 83, 109, 23);
		    bg.add(m_TestSplitBut);
		    
		    m_CVBut = new JRadioButton("Cross- validation");
		    m_CVBut.setBounds(20, 109, 109, 23);
		    bg.add(m_CVBut);
		    
		     m_PercentBut = new JRadioButton("Percentage split");
		     m_PercentBut.setBounds(20, 135, 109, 23);
		    bg.add(m_PercentBut);
		    
		    JButton btnSet = new JButton("Set...");
		    btnSet.setBounds(133, 83, 166, 24);
		    
		    JLabel lblFolds = new JLabel("Folds");
		    lblFolds.setBounds(156, 113, 30, 14);
		    
		    textField_Folds = new JTextField();
		    textField_Folds.setBounds(190, 110, 109, 20);
		    textField_Folds.setColumns(10);
		    
		    JLabel label = new JLabel("%");
		    label.setBounds(166, 139, 19, 14);
		    
		    textField_3 = new JTextField();
		    textField_3.setBounds(190, 136, 109, 20);
		    textField_3.setColumns(10);
		    
		    //ScrollPane e Tabela de sele��o de atributos do Classifier 1
		    JScrollPane scrollPaneClass1 = new JScrollPane();
		    scrollPaneClass1.setBounds(10, 183, 288, 151);
		    scrollPaneClass1.setBorder(new TitledBorder(null, "Select atributes", TitledBorder.LEADING, TitledBorder.TOP, null, null));		
		    
		    
		    
		    panelClass2 = new JPanel();
		    panelClass2.setBorder(new TitledBorder(UIManager.getBorder("TitledBorder.border"), "Classifier 2", TitledBorder.LEADING, TitledBorder.TOP, null, null));
		    
		    ButtonGroup bg2 = new ButtonGroup();  
		    
		    m_CVBut2 = new JRadioButton("Use traning set");
		    m_CVBut2.setBounds(20, 57, 109, 23);
		    m_CVBut2.setSelected(true);
		    bg2.add(m_CVBut2);
		    
		    m_TestSplitBut = new JRadioButton("Supplied test set");
		    m_TestSplitBut.setBounds(20, 83, 109, 23);
		    bg2.add(m_TestSplitBut);
		    
		    m_CVBut = new JRadioButton("Cross- validation");
		    m_CVBut.setBounds(20, 109, 109, 23);
		    bg2.add(m_CVBut);
		    
		    m_PercentBut = new JRadioButton("Percentage split");
		    m_PercentBut.setBounds(20, 135, 109, 23);
		    bg2.add(m_PercentBut);
		    
		    JButton button_1 = new JButton("Set...");
		    button_1.setBounds(133, 83, 166, 24);
		    
		    JLabel label_1 = new JLabel("Folds");
		    label_1.setBounds(156, 113, 30, 14);
		    
		    textField_Folds2 = new JTextField();
		    textField_Folds2.setBounds(190, 110, 109, 20);
		    textField_Folds2.setColumns(10);
		    
		    JLabel label_2 = new JLabel("%");
		    label_2.setBounds(166, 139, 19, 14);
		    
		    textField_5 = new JTextField();
		    textField_5.setBounds(190, 136, 109, 20);
		    textField_5.setColumns(10);
		    
		    JScrollPane scrollPane_1 = new JScrollPane();
		    tableClass2 = new JTable(modelClass2);
		    scrollPane_1.setViewportView(tableClass2);
		    
		    scrollPane_1.setBounds(10, 183, 288, 151);
		    scrollPane_1.setBorder(new TitledBorder(null, "Select atributes", TitledBorder.LEADING, TitledBorder.TOP, null, null));
		    //Tabela 		       		       
		    tableClass1 = new JTable(modelClass1);
		    scrollPaneClass1.setViewportView(tableClass1);
		    
		
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
					// Aqui chamaria um thread para execu��o do cootraining.
					
				
				}
			});
			m_StartBut.setBounds(10, 35, 152, 27);
			panelComfirm.add(m_StartBut);
			
			m_StopBut = new JButton("Stop");
			m_StopBut.setBounds(164, 35, 145, 27);
			panelComfirm.add(m_StopBut);
			
			JScrollPane scrollPaneResult = new JScrollPane();
			scrollPaneResult.setBounds(0, 73, 319, 137);
			panelComfirm.add(scrollPaneResult);
			scrollPaneResult.setBorder(new TitledBorder(null, "Result list (rigth click for options)", TitledBorder.LEADING, TitledBorder.TOP, null, null));
						
			scrollPaneResult.setViewportView(m_History);

		    
		    
		    GroupLayout gl_panelLeft = new GroupLayout(panelLeft);
		    gl_panelLeft.setHorizontalGroup(
		    	gl_panelLeft.createParallelGroup(Alignment.LEADING)
		    		.addComponent(panelClass1, GroupLayout.PREFERRED_SIZE, 319, GroupLayout.PREFERRED_SIZE)
		    		.addComponent(panelClass2, GroupLayout.PREFERRED_SIZE, 319, GroupLayout.PREFERRED_SIZE)
		    		.addComponent(panelComfirm, GroupLayout.PREFERRED_SIZE, 319, GroupLayout.PREFERRED_SIZE)
		    );
		    gl_panelLeft.setVerticalGroup(
		    	gl_panelLeft.createParallelGroup(Alignment.LEADING)
		    		.addGroup(gl_panelLeft.createSequentialGroup()
		    			.addComponent(panelClass1, GroupLayout.PREFERRED_SIZE, 345, GroupLayout.PREFERRED_SIZE)
		    			.addGap(6)
		    			.addComponent(panelClass2, GroupLayout.PREFERRED_SIZE, 345, GroupLayout.PREFERRED_SIZE)
		    			.addGap(6)
		    			.addComponent(panelComfirm, GroupLayout.PREFERRED_SIZE, 210, GroupLayout.PREFERRED_SIZE))
		    );
		    
		    panelClass2.setLayout(null);
		    panelClass2.add(m_TrainBut2);
		    panelClass2.add(m_TestSplitBut2);
		    panelClass2.add(m_CVBut2);
		    panelClass2.add(m_PercentBut2);
		    panelClass2.add(button_1);
		    panelClass2.add(label_1);
		    panelClass2.add(textField_Folds2);
		    panelClass2.add(label_2);
		    panelClass2.add(textField_5);
		    panelClass2.add(scrollPane_1);
		    
		    panelClass1.setLayout(null);
		    panelClass1.add(m_TrainBut);
		    panelClass1.add(m_TestSplitBut);
		    panelClass1.add(m_CVBut);
		    panelClass1.add(m_PercentBut);
		    panelClass1.add(btnSet);
		    panelClass1.add(lblFolds);
		    panelClass1.add(textField_Folds);
		    panelClass1.add(label);
		    panelClass1.add(textField_3);
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
		    
		    textField = new JTextField();
		    textField.setEditable(false);
		    textField.setBackground(Color.WHITE);
		    textField.setColumns(10);
		    
		    
		    GroupLayout gl_panel_3 = new GroupLayout(panel_3);
		    gl_panel_3.setHorizontalGroup(
		    	gl_panel_3.createParallelGroup(Alignment.LEADING)
		    		.addGroup(gl_panel_3.createSequentialGroup()
		    			.addGap(4)
		    			.addComponent(btnNewButton)
		    			.addGap(7)
		    			.addComponent(textField, GroupLayout.DEFAULT_SIZE, 465, Short.MAX_VALUE)
		    			.addGap(4))
		    );
		    gl_panel_3.setVerticalGroup(
		    	gl_panel_3.createParallelGroup(Alignment.LEADING)
		    		.addGroup(gl_panel_3.createSequentialGroup()
		    			.addGap(2)
		    			.addComponent(btnNewButton))
		    		.addGroup(gl_panel_3.createSequentialGroup()
		    			.addGap(3)
		    			.addComponent(textField, GroupLayout.PREFERRED_SIZE, 22, GroupLayout.PREFERRED_SIZE))
		    );
		    
		    
		    panel_3.setLayout(gl_panel_3);
		    btnNewButton.addMouseListener(new MouseAdapter() {
		    	
		    	public void mouseReleased(MouseEvent arg0) {
		    		//implementar aqui a escolhar do algoritmo self-traning ou traning. no entanto a interface n�o est� se adaptando a essa escolha.		
		    	    
		    	}
		    });
		    GroupLayout gl_panelP = new GroupLayout(panelP);
		    gl_panelP.setHorizontalGroup(
		    	gl_panelP.createParallelGroup(Alignment.LEADING)
		    		.addComponent(panel_3, GroupLayout.DEFAULT_SIZE, 647, Short.MAX_VALUE)
		    		.addGroup(gl_panelP.createSequentialGroup()
		    			.addComponent(scrollPane_2, GroupLayout.PREFERRED_SIZE, 336, GroupLayout.PREFERRED_SIZE)
		    			.addGap(6)
		    			.addComponent(scrollPane, GroupLayout.DEFAULT_SIZE, 305, Short.MAX_VALUE))
		    );
		    gl_panelP.setVerticalGroup(
		    	gl_panelP.createParallelGroup(Alignment.LEADING)
		    		.addGroup(gl_panelP.createSequentialGroup()
		    			.addComponent(panel_3, GroupLayout.PREFERRED_SIZE, 52, GroupLayout.PREFERRED_SIZE)
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
		// TODO Auto-generated method stub
		 m_Explorer = parent;
	}

	
	public Explorer getExplorer() {
		 return m_Explorer;
	}

	
	public void setInstances(Instances inst) {
		
		//Lyah, Aqui carrega o conjunto de dados escolhido pelo us�ario na primeira aba do weka , s� isso :).
	    m_Instances = inst;
	    
	    //colocando informa��es das intancias no texto de saida
	    m_OutText.setText(m_Instances.toString());
	    
	    //Adicionando os atributos do conjunto de dados nas tabelas de sele��o de atributos.
       for( int i= 0 ;i<m_Instances.numAttributes();i++){
    	  
    	   modelClass1.addRow(new Object[]{i+1, true, m_Instances.attribute(i).name()});
    	   modelClass2.addRow(new Object[]{i+1, true, m_Instances.attribute(i).name()});
       }
	    
       //Adicionando os atributos no combobox, para possibilitar para u us�ario a escolha do atributo classe.
       String[] attribNames = new String[m_Instances.numAttributes()];
       for (int i = 0; i < attribNames.length; i++) {
         String type = "(" + Attribute.typeToStringShort(m_Instances.attribute(i))
             + ") ";
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
		return "Semi-supervised";
	}

	
	public String getTabTitleToolTip() {
		return "Semi-supervised learning";
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
        System.out.println(aValue);
        Vector rowData = (Vector)getDataVector().get(row);
        rowData.set(1, (Boolean)aValue);
        fireTableCellUpdated(row, column);
      }
    }

  }


public void setLog(Logger newLog) {
	// TODO Auto-generated method stub
	
}
}



