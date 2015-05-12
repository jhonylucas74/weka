package weka.gui.explorer;

import java.awt.Color;
import java.awt.GridLayout;
import java.awt.event.InputEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.beans.PropertyChangeSupport;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Date;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.ListIterator;
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
import weka.clusterers.Clusterer;
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

import javax.swing.SwingConstants;
import javax.swing.JSpinner;
import javax.swing.SpinnerNumberModel;

import java.beans.PropertyEditor;

import javax.swing.BorderFactory;

import java.awt.BorderLayout;

import javax.swing.ScrollPaneConstants;

import java.awt.GridBagLayout;
import java.awt.GridBagConstraints;
import java.awt.Insets;
import java.awt.List;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;

public class SelfTrainingPanel extends JPanel implements ExplorerPanel,
		LogHandler {

	JSpinner spinAmount = new JSpinner();

	/** Essa vari�vel mant�m a conex�o com o Explorer do weka */
	protected Explorer m_Explorer = null;

	/** Conjunto de dados carregado originalmente pelo usu�rio */
	private Instances m_Instances;

	/** Panel de texto(textPane) onde o resultado � mostrado para o usu�rio */
	private JTextPane m_OutText = new JTextPane();

	/**
	 * Jpanel do lado esquerdo onde fica os outros componentes com as
	 * configura��es do classificador
	 */
	private JPanel panelLeft;

	/** Jpanel do classificador */
	private JPanel panelClassifier;

	private JRadioButton rdbtnTraditional;
	private JRadioButton rdbtnExtensionWithCluster;

	/** bot�es radio para sele��o do modo de treinamento do classificador */
	private ButtonGroup buttongroup;
	private JRadioButton radioCV;
	private JRadioButton radioTrainSet;

	private JPanel panelClustererSettings;
	private JPanel panelChooseClusterer;
	private JLabel lblIncorporationMode;

	private JSpinner spnPorcetageUnlabeled;

	private JRadioButton balanced;
	private JRadioButton relative;

	/**
	 * Combo box respons�vel pela escolha do atributo class do conjunto de dados
	 */
	private JComboBox comboBox;

	/** Lets the user configure the classifier. */
	public static GenericObjectEditor m_ClassifierEditor = new GenericObjectEditor();

	protected static GenericObjectEditor m_ClustererEditor = new GenericObjectEditor();

	/** The panel showing the current classifier selection. */
	protected PropertyPanel m_ClassifierEPanel = new PropertyPanel(
			m_ClassifierEditor);

	protected PropertyPanel m_ClustererEPanel = new PropertyPanel(
			m_ClustererEditor);

	protected ResultHistoryPanel m_History = new ResultHistoryPanel(m_OutText);

	/** JButton para iniciar classifica��o. */
	protected JButton m_StartBut = new JButton("Start");

	/** JButton para interromper classifica��o */
	protected JButton m_StopBut = new JButton("Stop");

	/** The destination for log/status messages. */
	protected Logger m_Log = new SysErrLog();
	private final ButtonGroup buttonGroup = new ButtonGroup();
	private final ButtonGroup buttonGroup_1 = new ButtonGroup();
	private JSpinner spinFolds;

	/**
	 * @wbp.parser.entryPoint
	 */

	public SelfTrainingPanel() {

		setLayout(new GridLayout());

		JPanel panelSelfTraining = new JPanel();

		add(panelSelfTraining);

		m_ClassifierEditor.setClassType(Classifier.class);
		m_ClassifierEditor.setValue(ExplorerDefaults.getClassifier());
		m_ClassifierEditor
				.addPropertyChangeListener(new PropertyChangeListener() {
					public void propertyChange(PropertyChangeEvent e) {

						// Check capabilities
						Capabilities currentFilter = m_ClassifierEditor
								.getCapabilitiesFilter();
						Classifier classifier = (Classifier) m_ClassifierEditor
								.getValue();

						Capabilities currentSchemeCapabilities = null;
						if (classifier != null && currentFilter != null
								&& (classifier instanceof CapabilitiesHandler)) {
							currentSchemeCapabilities = ((CapabilitiesHandler) classifier)
									.getCapabilities();

							if (!currentSchemeCapabilities
									.supportsMaybe(currentFilter)
									&& !currentSchemeCapabilities
											.supports(currentFilter)) {

							}
						}

						repaint();
					}
				});

		m_ClustererEditor.setClassType(Clusterer.class);
		m_ClustererEditor.setValue(ExplorerDefaults.getClusterer());
		m_ClustererEditor
				.addPropertyChangeListener(new PropertyChangeListener() {
					public void propertyChange(PropertyChangeEvent e) {
						m_StartBut.setEnabled(true); // Aqui � para ativar o
														// bot�o start ,
														// verifique se o nome
														// do atributo � o mesmo
														// na clase
														// SelfTrainingPanel
						Capabilities currentFilter = m_ClustererEditor
								.getCapabilitiesFilter();
						Clusterer clusterer = (Clusterer) m_ClustererEditor
								.getValue();

						Capabilities currentSchemeCapabilities = null;
						if (clusterer != null && currentFilter != null
								&& (clusterer instanceof CapabilitiesHandler)) {
							currentSchemeCapabilities = ((CapabilitiesHandler) clusterer)
									.getCapabilities();

							if (!currentSchemeCapabilities
									.supportsMaybe(currentFilter)
									&& !currentSchemeCapabilities
											.supports(currentFilter)) {
								m_StartBut.setEnabled(false); // Aqui da mesma
																// forma do
																// ultimo
																// coment�rio,
																// caso n�o
																// reconhe�a o
																// m_startBut
																// verifique
																// como est�
																// sendo chamado
																// no Panel.
							}
						}
						repaint();
					}
				});

		buttongroup = new ButtonGroup();
		GridBagLayout gbl_panelSelfTraining = new GridBagLayout();
		gbl_panelSelfTraining.columnWidths = new int[] { 331, 497, 0 };
		gbl_panelSelfTraining.rowHeights = new int[] { 52, 406, 165, 0 };
		gbl_panelSelfTraining.columnWeights = new double[] { 0.0, 1.0,
				Double.MIN_VALUE };
		gbl_panelSelfTraining.rowWeights = new double[] { 0.0, 1.0, 1.0,
				Double.MIN_VALUE };
		panelSelfTraining.setLayout(gbl_panelSelfTraining);

		JScrollPane scrollPaneResult = new JScrollPane();
		scrollPaneResult.setBorder(new TitledBorder(null,
				"Result list (rigth click for options)", TitledBorder.LEADING,
				TitledBorder.TOP, null, null));

		scrollPaneResult.setViewportView(m_History);

		m_History.setHandleRightClicks(false);
		// see if we can popup a menu for the selected result
		m_History.getList().addMouseListener(new MouseAdapter() {
			@Override
			public void mouseClicked(MouseEvent e) {
				if (((e.getModifiers() & InputEvent.BUTTON1_MASK) != InputEvent.BUTTON1_MASK)
						|| e.isAltDown()) {
					int index = m_History.getList().locationToIndex(
							e.getPoint());
					if (index != -1) {
						String name = m_History.getNameAtIndex(index);

					} else {

					}
				}
			}
		});

		JPanel panelAlgorithmVersion = new JPanel();
		panelAlgorithmVersion.addMouseListener(new MouseAdapter() {
			@Override
			public void mouseClicked(MouseEvent arg0) {

			}
		});
		panelAlgorithmVersion.setBorder(new TitledBorder(UIManager
				.getBorder("TitledBorder.border"), "Algorithm version",
				TitledBorder.LEADING, TitledBorder.TOP, null, null));

		rdbtnTraditional = new JRadioButton("Traditional");
		rdbtnTraditional.setSelected(true);
		rdbtnTraditional.addMouseListener(new MouseAdapter() {
			@Override
			public void mouseClicked(MouseEvent arg0) {
				if (rdbtnTraditional.isSelected()) {
					panelClustererSettings.setEnabled(false);
					panelChooseClusterer.setEnabled(false);
					m_ClustererEPanel.setEnabled(false);
					lblIncorporationMode.setEnabled(false);
					balanced.setEnabled(false);
					relative.setEnabled(false);
				}
			}
		});

		rdbtnTraditional.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent arg0) {

			}
		});
		buttonGroup_1.add(rdbtnTraditional);

		rdbtnExtensionWithCluster = new JRadioButton("Extension with cluster");
		rdbtnExtensionWithCluster.addMouseListener(new MouseAdapter() {
			@Override
			public void mouseClicked(MouseEvent arg0) {
				if (rdbtnExtensionWithCluster.isSelected()) {
					panelClustererSettings.setEnabled(true);
					panelChooseClusterer.setEnabled(true);
					m_ClustererEPanel.setEnabled(true);
					lblIncorporationMode.setEnabled(true);
					balanced.setEnabled(true);
					relative.setEnabled(true);
				}
			}
		});
		buttonGroup_1.add(rdbtnExtensionWithCluster);

		GroupLayout gl_panelAlgorithmVersion = new GroupLayout(
				panelAlgorithmVersion);
		gl_panelAlgorithmVersion.setHorizontalGroup(gl_panelAlgorithmVersion
				.createParallelGroup(Alignment.LEADING).addGroup(
						gl_panelAlgorithmVersion.createSequentialGroup()
								.addGap(60).addComponent(rdbtnTraditional)
								.addGap(18)
								.addComponent(rdbtnExtensionWithCluster)
								.addContainerGap(309, Short.MAX_VALUE)));
		gl_panelAlgorithmVersion
				.setVerticalGroup(gl_panelAlgorithmVersion
						.createParallelGroup(Alignment.LEADING)
						.addGroup(
								gl_panelAlgorithmVersion
										.createSequentialGroup()
										.addGroup(
												gl_panelAlgorithmVersion
														.createParallelGroup(
																Alignment.BASELINE)
														.addComponent(
																rdbtnTraditional)
														.addComponent(
																rdbtnExtensionWithCluster))
										.addContainerGap(
												GroupLayout.DEFAULT_SIZE,
												Short.MAX_VALUE)));

		panelAlgorithmVersion.setLayout(gl_panelAlgorithmVersion);
		GridBagConstraints gbc_panelAlgorithmVersion = new GridBagConstraints();
		gbc_panelAlgorithmVersion.fill = GridBagConstraints.HORIZONTAL;
		gbc_panelAlgorithmVersion.insets = new Insets(0, 0, 5, 0);
		gbc_panelAlgorithmVersion.gridwidth = 2;
		gbc_panelAlgorithmVersion.gridx = 0;
		gbc_panelAlgorithmVersion.gridy = 0;
		panelSelfTraining.add(panelAlgorithmVersion, gbc_panelAlgorithmVersion);

		panelLeft = new JPanel();
		GridBagConstraints gbc_panelLeft = new GridBagConstraints();
		gbc_panelLeft.fill = GridBagConstraints.BOTH;
		gbc_panelLeft.insets = new Insets(0, 0, 5, 5);
		gbc_panelLeft.gridx = 0;
		gbc_panelLeft.gridy = 1;
		panelSelfTraining.add(panelLeft, gbc_panelLeft);
		panelClassifier = new JPanel();
		panelClassifier.setBounds(0, 0, 326, 317);
		panelClassifier.setBorder(new TitledBorder(UIManager
				.getBorder("TitledBorder.border"), "Classifier ",
				TitledBorder.LEADING, TitledBorder.TOP, null, null));

		radioTrainSet = new JRadioButton("Use traning set");
		radioTrainSet.setBounds(20, 57, 109, 23);
		radioTrainSet.setSelected(true);
		buttongroup.add(radioTrainSet);

		radioCV = new JRadioButton("Cross- validation");
		radioCV.setBounds(20, 83, 109, 23);
		buttongroup.add(radioCV);

		JLabel lblFolds = new JLabel("Folds");
		lblFolds.setBounds(220, 83, 30, 23);

		JPanel panelComfirm = new JPanel();
		panelComfirm.setBounds(0, 322, 319, 73);
		panelComfirm.setBorder(null);
		add(panelComfirm);
		panelComfirm.setLayout(null);

		comboBox = new JComboBox();
		comboBox.setBounds(0, 0, 319, 34);
		panelComfirm.add(comboBox);

		m_StartBut = new JButton("Start");
		m_StartBut.addMouseListener(new MouseAdapter() {
			@Override
			public void mouseReleased(MouseEvent arg0) {
				ArrayList<Double> listaPctCorreto = new ArrayList<Double>();
				double media;
				double desvioPadrao;

				for (int i = 0; i < 10; i++) {
					/*
					 * Inst�nciando um objeto do tipo SelfTraining. � possiv�l
					 * fazer isso de duas maneiras: Passando os par�metros pelo
					 * construtor. Definindo os valores.
					 */
					SelfTraining selfTraining = new SelfTraining();
					selfTraining.setSetOriginal(m_Instances); // Definindo o
																// conjunto
																// original
					selfTraining
							.setAlgoClassificacao((Classifier) m_ClassifierEditor
									.getValue());// Definindo o classificador
													// escolhido pelo us�ario
					selfTraining.setClassIndex(comboBox.getSelectedIndex());// Definindo
																			// o
																			// atributo
																			// alvo
																			// escolhido
																			// pelo
																			// usu�rio
					selfTraining.setM_OutText(m_OutText);

					double amount = (Double) spinAmount.getValue();
					selfTraining.setPercent(amount / 100);

					if (rdbtnExtensionWithCluster.isSelected()) {
						selfTraining.setClustererVersion(true);
						selfTraining
								.setAlgoAgrupamento((Clusterer) m_ClustererEditor
										.getValue());
						selfTraining.setIncorporationMode(balanced.isSelected());
					}

					/* Capturando o metodo de agrupamento escolhido pelo usuario */
					/* Descobrindo qual dos bot�es radio est� selecionado */
					int modoTreino = 0;

					Enumeration<AbstractButton> allRadioButton = buttongroup
							.getElements();
					while (allRadioButton.hasMoreElements()) {
						JRadioButton temp = (JRadioButton) allRadioButton
								.nextElement();
						if (temp.isSelected()) {
							if (temp.getText().equals("Use traning set")) {
								modoTreino = 0;
							}

							if (temp.getText().equals("Supplied test set")) {
								modoTreino = 1;
							}

							if (temp.getText().equals("Cross- validation")) {
								modoTreino = 2;
							}

							if (temp.getText().equals("Percentage split")) {
								modoTreino = 3;
							}
						}
					}

					if (radioCV.isSelected()) {
						selfTraining.setModoTreino(modoTreino);
						int numFolds = (Integer) spinFolds.getValue();
						selfTraining.setFolds(numFolds);
					}

					int percentunlabeled = (Integer) spnPorcetageUnlabeled
							.getValue();

					if (percentunlabeled <= 0 || percentunlabeled > 100) {
						percentunlabeled = 10;
					}
					selfTraining.setPercentageUnlabeled(percentunlabeled);
					selfTraining.setLog(m_Log);
					selfTraining.run();

					listaPctCorreto.add(selfTraining.getPctCorreto());

					// Thread executable = new Thread(selfTraining);
					// executable.start();
				}

				double soma = 0;
				for (int i = 0; i < listaPctCorreto.size(); i++)
					soma += listaPctCorreto.get(i);
				media = soma / 10;

				desvioPadrao = 0;
				for (int i = 0; i < listaPctCorreto.size(); i++)
					desvioPadrao += Math.pow(listaPctCorreto.get(i) - media, 2);
				desvioPadrao = Math.sqrt(desvioPadrao / 9);

				SelfTraining.m_OutText.setText("Olaaaaaaaaaaaaaa");

				DecimalFormat fmt = new DecimalFormat("0.00");
				String mediaStr = fmt.format(media);
				String desvioPadraoStr = fmt.format(desvioPadrao);

				SelfTraining.m_OutText
						.setText("Inst�ncias classificadas corretamente \n\n"
								+ "Percentual m�dio: " + mediaStr
								+ "\n Desvio padr�o: " + desvioPadraoStr);
			}
		});
		m_StartBut.setBounds(10, 35, 152, 27);
		panelComfirm.add(m_StartBut);

		m_StopBut = new JButton("Stop");
		m_StopBut.setBounds(164, 35, 145, 27);
		panelComfirm.add(m_StopBut);
		panelLeft.setLayout(null);

		panelClassifier.setLayout(null);
		panelClassifier.add(radioTrainSet);
		panelClassifier.add(radioCV);
		panelClassifier.add(lblFolds);

		JPanel panelChooseClassifier = new JPanel();
		panelChooseClassifier.setBounds(10, 21, 296, 29);
		panelChooseClassifier.setLayout(null);
		m_ClassifierEPanel.setBounds(10, 5, 276, 23);
		panelChooseClassifier.add(m_ClassifierEPanel);

		panelClassifier.add(panelChooseClassifier);
		panelLeft.add(panelClassifier);

		JPanel panelBestSamples = new JPanel();
		panelBestSamples.setBounds(10, 143, 296, 50);
		panelClassifier.add(panelBestSamples);
		panelBestSamples.setBorder(new TitledBorder(UIManager
				.getBorder("TitledBorder.border"), "Best samples",
				TitledBorder.LEADING, TitledBorder.TOP, null, null));
		panelBestSamples.setLayout(null);

		JLabel lblBestSamples = new JLabel("Incorporate per cycle");
		lblBestSamples.setBounds(10, 24, 103, 14);
		panelBestSamples.add(lblBestSamples);

		JLabel label = new JLabel("%");
		label.setHorizontalAlignment(SwingConstants.TRAILING);
		label.setBounds(267, 24, 19, 14);
		panelBestSamples.add(label);

		spinAmount.setModel(new SpinnerNumberModel(10, 1, 100, 5));
		spinAmount.setBounds(223, 21, 47, 20);
		panelBestSamples.add(spinAmount);

		panelClustererSettings = new JPanel();
		panelClustererSettings.setBounds(10, 196, 296, 110);
		panelClassifier.add(panelClustererSettings);
		panelClustererSettings.setBorder(new TitledBorder(UIManager
				.getBorder("TitledBorder.border"), "Clusterer settings",
				TitledBorder.LEADING, TitledBorder.TOP, null, null));
		panelClustererSettings.setLayout(null);

		panelChooseClusterer = new JPanel();
		panelChooseClusterer.setLayout(null);
		panelChooseClusterer.setBounds(10, 21, 276, 29);
		panelClustererSettings.add(panelChooseClusterer);

		m_ClustererEPanel.setBounds(10, 5, 256, 23);
		panelChooseClusterer.add(m_ClustererEPanel);

		balanced = new JRadioButton("Balanced");
		balanced.setSelected(true);
		balanced.setEnabled(false);
		balanced.setBounds(53, 80, 69, 23);
		panelClustererSettings.add(balanced);
		buttonGroup.add(balanced);

		relative = new JRadioButton("Relative");
		relative.setEnabled(false);
		relative.setBounds(152, 80, 65, 23);
		panelClustererSettings.add(relative);
		buttonGroup.add(relative);

		lblIncorporationMode = new JLabel("Incorporation mode:");
		lblIncorporationMode.setEnabled(false);
		lblIncorporationMode.setBounds(10, 61, 129, 14);
		panelClustererSettings.add(lblIncorporationMode);

		JLabel lblUnlabeledInstances = new JLabel("Unlabeled Instances");
		lblUnlabeledInstances.setBounds(20, 113, 109, 14);
		panelClassifier.add(lblUnlabeledInstances);

		JLabel label_1 = new JLabel("%");
		label_1.setHorizontalAlignment(SwingConstants.TRAILING);
		label_1.setBounds(189, 113, 11, 14);
		panelClassifier.add(label_1);

		spnPorcetageUnlabeled = new JSpinner();
		spnPorcetageUnlabeled.setModel(new SpinnerNumberModel(10, 1, 100, 5));
		spnPorcetageUnlabeled.setBounds(134, 110, 46, 20);
		panelClassifier.add(spnPorcetageUnlabeled);
		panelLeft.add(panelComfirm);

		JScrollPane scrollPane = new JScrollPane();
		scrollPane
				.setVerticalScrollBarPolicy(ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS);
		scrollPane
				.setHorizontalScrollBarPolicy(ScrollPaneConstants.HORIZONTAL_SCROLLBAR_ALWAYS);
		scrollPane.setBorder(new TitledBorder(UIManager
				.getBorder("TitledBorder.border"), "SelfTraining output",
				TitledBorder.LEADING, TitledBorder.TOP, null, null));
		m_OutText.setFont(new Font("Monospaced", Font.PLAIN, 12));
		scrollPane.setViewportView(m_OutText);
		GridBagConstraints gbc_scrollPane = new GridBagConstraints();
		gbc_scrollPane.fill = GridBagConstraints.BOTH;
		gbc_scrollPane.gridheight = 2;
		gbc_scrollPane.gridx = 1;
		gbc_scrollPane.gridy = 1;
		panelSelfTraining.add(scrollPane, gbc_scrollPane);
		GridBagConstraints gbc_scrollPaneResult = new GridBagConstraints();
		gbc_scrollPaneResult.fill = GridBagConstraints.BOTH;
		gbc_scrollPaneResult.insets = new Insets(0, 0, 0, 5);
		gbc_scrollPaneResult.gridx = 0;
		gbc_scrollPaneResult.gridy = 2;
		panelSelfTraining.add(scrollPaneResult, gbc_scrollPaneResult);

		panelClustererSettings.setEnabled(false);
		panelChooseClusterer.setEnabled(false);
		m_ClustererEPanel.setEnabled(false);
		lblIncorporationMode.setEnabled(false);
		balanced.setEnabled(false);
		relative.setEnabled(false);

		spinFolds = new JSpinner();
		spinFolds.setModel(new SpinnerNumberModel(10, 1, 100, 1));
		spinFolds.setBounds(260, 84, 46, 20);
		panelClassifier.add(spinFolds);
	}

	private void setClustererEnable() {

	}

	private void setClustererDisable() {

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

		// Adicionando os atributos no combobox, para possibilitar para u
		// us�ario a escolha do atributo classe.
		String[] attribNames = new String[m_Instances.numAttributes()];
		for (int i = 0; i < attribNames.length; i++) {
			String type = "("
					+ Attribute.typeToStringShort(m_Instances.attribute(i))
					+ ") ";
			attribNames[i] = type + m_Instances.attribute(i).name();
		}

		comboBox.setModel(new DefaultComboBoxModel(attribNames));

		// dizendo pro comboBox que por padr�o o atributo final ser� o atributo
		// class
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

	public class MyTableModel extends DefaultTableModel {

		public MyTableModel() {
			super(new String[] { "Num.", " ", "Atribute" }, 0);
		}

		@Override
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

		@Override
		public boolean isCellEditable(int row, int column) {
			return column == 1;
		}

		@Override
		public void setValueAt(Object aValue, int row, int column) {
			if (aValue instanceof Boolean && column == 1) {
				System.out.println(aValue);
				Vector rowData = (Vector) getDataVector().get(row);
				rowData.set(1, (Boolean) aValue);
				fireTableCellUpdated(row, column);
			}
		}

	}

	public void setLog(Logger newLog) {
		// TODO Auto-generated method stub

	}
}
