package weka.classifiers.trees.randomForestSplitCriterionSelection;

import weka.core.ContingencyTables;

/**
 * Standar Mutual Information or Information Gain
 * splitting criterion. 
 * 
 * @author Simone Romano sromano@student.unimelb.edu.au
 *
 */
public class MISplit extends Split{
	
	/**
	 * Class Entropy
	 */
	protected double m_priorVal;
	
	/**
	 * It saves a reference of the 
	 * contingency table and the class
	 * entropy in order to avoid to compute it
	 * later. 
	 * 
	 * @param matrix contingency table
	 */
	public MISplit(double[][] matrix){
		m_matrix = matrix;
		m_priorVal = ContingencyTables.entropyOverColumns(m_matrix);			  
	}
	
	/**
	 * Return the Mutual Information or the the Information
	 * Gain for the contingency table.
	 */
	public double gain(){
		return m_priorVal - ContingencyTables.entropyConditionedOnRows(m_matrix);
	}
}
