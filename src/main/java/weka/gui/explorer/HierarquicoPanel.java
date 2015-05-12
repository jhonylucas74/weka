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
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import javax.swing.AbstractButton;
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
import weka.learning.semisupervised.SelfTraining;




import javax.swing.LayoutStyle.ComponentPlacement;

import java.awt.Font;
import javax.swing.SpringLayout;
import java.awt.Rectangle;

public class HierarquicoPanel extends JPanel  implements ExplorerPanel, LogHandler{

	 /** Essa vari�vel mant�m a conex�o com o Explorer do weka */
	 protected Explorer m_Explorer = null;
	 
	 /** Conjunto de dados carregado originalmente pelo usu�rio */
	 private Instances m_Instances;		
	 
	 /** Panel de texto(textPane) onde o resultado � mostrado para o usu�rio*/
	 private JTextPane m_OutText = new JTextPane();	 
	
	 /** Jpanel do classificador*/
	 private  JPanel panelClass1;
	 
	 /** bot�es radio para sele��o do modo de treinamento do classificador */
	 private ButtonGroup buttongroup ;
	 private JRadioButton m_CVBut;
	 private JRadioButton m_PercentBut;
	 private JRadioButton m_TrainBut;
	 private JRadioButton m_TestSplitBut;	 
	 
	 /** n�meros de folds do classificador */
	 private JTextField textField_Folds;
	 
	 /** split porcetagem do classificador */
	 private JTextField textField_split;
	 	 
	 /** Combo box respons�vel pela escolha do atributo class do conjunto de dados*/
	 private JComboBox comboBox;
	 
	 /** Lets the user configure the classifier. */
     public static GenericObjectEditor m_ClassifierEditor = new GenericObjectEditor();
	
	 /** The panel showing the current classifier selection. */
	 protected PropertyPanel m_CEPanel = new PropertyPanel(m_ClassifierEditor);
	 	 
	 protected ResultHistoryPanel m_History = new ResultHistoryPanel(m_OutText);
	  	  
	 /** JButton para iniciar classifica��o. */
	 protected JButton m_StartBut = new JButton("Start");
	
	 /** JButton para interromper classifica��o */
	 protected JButton m_StopBut = new JButton("Stop");
	  
	 /** The destination for log/status messages. */
	 protected Logger m_Log = new SysErrLog();	  

	 /** ComboBox dos m�todos Hierarquico */
	 JComboBox comboMH ;
	 
	 /** Checkbox */
	 JCheckBox chckbxPredioObrigatriaNas = new JCheckBox("Binding prediction in the leaves");
	  
	  /**
		  * @wbp.parser.entryPoint
		  */
		  
	 public HierarquicoPanel(){
		    
		    
		    setLayout(new GridLayout()); 
		    
		    JPanel panelP = new JPanel();			   
		    add(panelP);
		    		    		   
		    
		    JScrollPane scrollPane = new JScrollPane();
		    scrollPane.setBorder(new TitledBorder(UIManager.getBorder("TitledBorder.border"), "Output", TitledBorder.LEADING, TitledBorder.TOP, null, null));
		    m_OutText.setFont(new Font("Monospaced", Font.PLAIN, 12));
		    scrollPane.setViewportView(m_OutText);
		    
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
		    
		    		   
		    
		    buttongroup = new ButtonGroup();  
		    
		    JScrollPane scrollPaneResult = new JScrollPane();
		    scrollPaneResult.setBorder(new TitledBorder(null, "Result list (rigth click for options)", TitledBorder.LEADING, TitledBorder.TOP, null, null));
		    
			scrollPaneResult.setViewportView(m_History);
			
			
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
		    panelClass1 = new JPanel();
		    panelClass1.setBounds(new Rectangle(0, 0, 200, 200));
		    panelClass1.setBorder(new TitledBorder(UIManager.getBorder("TitledBorder.border"), "Classifier ", TitledBorder.LEADING, TitledBorder.TOP, null, null));
		    
		    m_TrainBut = new JRadioButton("Use traning set");
		    m_TrainBut.setBounds(20, 57, 109, 23);
		    m_TrainBut.setSelected(true);
		    buttongroup.add(m_TrainBut);
		    
		    m_TestSplitBut = new JRadioButton("Supplied test set");
		    m_TestSplitBut.setBounds(20, 83, 109, 23);
		    buttongroup.add(m_TestSplitBut);
		    
		    m_CVBut = new JRadioButton("Cross- validation");
		    m_CVBut.setBounds(20, 109, 109, 23);
		    buttongroup.add(m_CVBut);
		    
		    m_PercentBut = new JRadioButton("Percentage split");
		    m_PercentBut.setBounds(20, 135, 109, 23);
		    buttongroup.add(m_PercentBut);
		    
		    JButton btnSet = new JButton("Set...");
		    btnSet.setBounds(133, 83, 166, 24);
		    
		    JLabel lblFolds = new JLabel("Folds");
		    lblFolds.setBounds(156, 113, 30, 14);
		    
		    textField_Folds = new JTextField();
		    textField_Folds.setBounds(190, 110, 109, 20);
		    textField_Folds.setColumns(10);
		    
		    JLabel label = new JLabel("%");
		    label.setBounds(166, 139, 19, 14);
		    
		    textField_split = new JTextField();
		    textField_split.setBounds(190, 136, 109, 20);
		    textField_split.setColumns(10);
		    
		    panelClass1.setLayout(null);
		    panelClass1.add(m_TrainBut);
		    panelClass1.add(m_TestSplitBut);
		    panelClass1.add(m_CVBut);
		    panelClass1.add(m_PercentBut);
		    panelClass1.add(btnSet);
		    panelClass1.add(lblFolds);
		    panelClass1.add(textField_Folds);
		    panelClass1.add(label);
		    panelClass1.add(textField_split);
		    
		    JPanel panel = new JPanel();
		    panel.setBounds(10, 21, 299, 29);
		    panel.setLayout(null);
		    m_CEPanel.setBounds(10, 5, 289, 23);
		    panel.add(m_CEPanel);
		    
		    panelClass1.add(panel);
		    
		
		    JPanel panelComfirm = new JPanel();
		    panelComfirm.setBounds(new Rectangle(0, 0, 200, 200));
		    panelComfirm.setBorder(null);
		    add(panelComfirm);
		    panelComfirm.setLayout(null);
		    
		    comboBox = new JComboBox();
		    comboBox.setBounds(10, 0, 309, 34);
		    panelComfirm.add(comboBox);
		    
		    m_StartBut = new JButton("Start");
		    m_StartBut.addMouseListener(new MouseAdapter() {
		    	
		    	public void mouseReleased(MouseEvent arg0) {					   
		    		
		 
		    			
		    /* Descobrindo qual dos bot�es radio est� selecionado  */
		    int modoTreino= 0;
		    boolean treinarFolhas =  chckbxPredioObrigatriaNas.isSelected();
		    int qualClasse = comboBox.getSelectedIndex();
		    int metodoH = comboMH.getSelectedIndex();;
		    
		    
	        Enumeration<AbstractButton> allRadioButton=buttongroup.getElements();  
	        while(allRadioButton.hasMoreElements())  
	        {  
	           JRadioButton temp=(JRadioButton)allRadioButton.nextElement();  
	           if(temp.isSelected())  
	           {  
		             if(temp.getText().equals("Use traning set")){
		            	 modoTreino = 0;
		    	     }
		             
		             if(temp.getText().equals("Supplied test set")){
		            	 modoTreino = 1;
		    	     }
		             
		             if(temp.getText().equals("Cross- validation")){
		            	 modoTreino = 2;
		    	     }
		             
		             if(temp.getText().equals("Percentage split")){
		            	 modoTreino = 3;
		    	     }
		             
	           }  
	        }            
		    
	       /**
	        * Aqui chama sua implementa��o
	        */
		      			  				
		    	}
		    });
		    m_StartBut.setBounds(10, 105, 152, 27);
		    panelComfirm.add(m_StartBut);
		    
		    m_StopBut = new JButton("Stop");
		    m_StopBut.setBounds(164, 105, 145, 27);
		    panelComfirm.add(m_StopBut);
		    
		    comboMH = new JComboBox();
		    comboMH.setBounds(10, 40, 309, 34);
		    panelComfirm.add(comboMH);
		    
		    chckbxPredioObrigatriaNas = new JCheckBox("Binding prediction in the leaves");
		    
		    
		    chckbxPredioObrigatriaNas.setBounds(10, 79, 252, 23);
		    panelComfirm.add(chckbxPredioObrigatriaNas);
		    GroupLayout gl_panelP = new GroupLayout(panelP);
		    gl_panelP.setHorizontalGroup(
		    	gl_panelP.createParallelGroup(Alignment.LEADING)
		    		.addGroup(gl_panelP.createSequentialGroup()
		    			.addContainerGap()
		    			.addGroup(gl_panelP.createParallelGroup(Alignment.TRAILING, false)
		    				.addComponent(scrollPaneResult, GroupLayout.PREFERRED_SIZE, 319, GroupLayout.PREFERRED_SIZE)
		    				.addGroup(Alignment.LEADING, gl_panelP.createParallelGroup(Alignment.TRAILING, false)
		    					.addComponent(panelComfirm, Alignment.LEADING, 0, 0, Short.MAX_VALUE)
		    					.addComponent(panelClass1, Alignment.LEADING, GroupLayout.DEFAULT_SIZE, 323, Short.MAX_VALUE)))
		    			.addPreferredGap(ComponentPlacement.RELATED)
		    			.addComponent(scrollPane, GroupLayout.DEFAULT_SIZE, 386, Short.MAX_VALUE)
		    			.addContainerGap())
		    );
		    gl_panelP.setVerticalGroup(
		    	gl_panelP.createParallelGroup(Alignment.LEADING)
		    		.addGroup(Alignment.TRAILING, gl_panelP.createSequentialGroup()
		    			.addGap(23)
		    			.addGroup(gl_panelP.createParallelGroup(Alignment.TRAILING)
		    				.addComponent(scrollPane, Alignment.LEADING, GroupLayout.DEFAULT_SIZE, 432, Short.MAX_VALUE)
		    				.addGroup(gl_panelP.createSequentialGroup()
		    					.addComponent(panelClass1, GroupLayout.PREFERRED_SIZE, 175, GroupLayout.PREFERRED_SIZE)
		    					.addPreferredGap(ComponentPlacement.RELATED)
		    					.addComponent(panelComfirm, GroupLayout.PREFERRED_SIZE, 145, GroupLayout.PREFERRED_SIZE)
		    					.addPreferredGap(ComponentPlacement.UNRELATED)
		    					.addComponent(scrollPaneResult, GroupLayout.DEFAULT_SIZE, 83, Short.MAX_VALUE)))
		    			.addContainerGap())
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
		
		// Atribuindo a m_instances o conjunto de dados escolhido pelo usu�rio. 
	    m_Instances = inst;
	    	
       //Adicionando os atributos no combobox, para possibilitar para u us�ario a escolha do atributo classe.
       String[] attribNames = new String[m_Instances.numAttributes()];
       for (int i = 0; i < attribNames.length; i++) {
         String type = "(" + Attribute.typeToStringShort(m_Instances.attribute(i))
             + ") ";
         attribNames[i] = type + m_Instances.attribute(i).name();
       }
       
       comboBox.setModel(new DefaultComboBoxModel(attribNames));
       
       //ComboMH 
       String[] modoH = new String[3];//Modo H classificadores hierarquicos
       modoH[0]="HMLP";
       modoH[1]="HMC";
       modoH[2]="HBR";
       
       comboMH.setModel(new DefaultComboBoxModel(modoH));
       
       
    
       
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
		return "Self-training";
	}

	
	public String getTabTitleToolTip() {
		return "Self-training learning";
	}	 
	  
	
	public void printOut(String out){
		
		m_OutText.setText(out);				
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



