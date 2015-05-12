package weka.classifiers.trees.randomForestSplitCriterionSelection;

import weka.core.Instance;
import weka.core.UnassignedClassException;

/**
 * Abstract class for a split. It stores the contingency table
 * for a given split.
 *  
 * @author Simone Romano (sromano@student.unimelb.edu.au)
 *
 */
public abstract class Split {
	
	/**
	 * Contingency table reference
	 */
	double[][] m_matrix;
	
	/**
	 * It returns the gain for a given split
	 */
	public double gain(){
		return Double.NEGATIVE_INFINITY;
	}
	
	/**
	 * It shift an instance from one bag to 
	 * another one.   
	 * 
	 * @param from the bag to shift from
	 * @param to the bag to shift in
	 * @param instance the instance to shift
	 * @throws Exception It throws UnassingedClassException
	 * 
	 */	
	public void shift(int from, int to, Instance instance) throws UnassignedClassException{
    int classIndex;
    double weight;
    classIndex = (int)instance.classValue();
    weight = instance.weight();
    m_matrix[from][classIndex] -= weight;
    m_matrix[to][classIndex] += weight;
	}
	
	/**
	 * It return a perfect copy of the
	 * contingency table.
	 * 
	 * @return copy of contingency_table
	 */
	public double[][] getCopyOfContingencyTable(){
    double[][] cp_matrix = new double[m_matrix.length][m_matrix[0].length]; 
    for (int j = 0; j < m_matrix.length; j++)
      System.arraycopy(m_matrix[j], 0, cp_matrix[j], 0, cp_matrix[j].length);
    return cp_matrix;
	}
	
}
