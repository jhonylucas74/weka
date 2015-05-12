package weka.classifiers.trees.randomForestSplitCriterionSelection;

import weka.core.ContingencyTables;
import weka.core.Utils;

import org.apache.commons.math3.special.Gamma;

/**
 * All entropies in this class are computed
 * with the Grassberger estimator. Remark: it gives 
 * an estimation of entropy as if it was computed with
 * natural logarithm. 
 * 
 * @author Simone Romano
 *
 */

public class GrassbergerMISplit extends Split{
	
	protected double m_priorVal;
	
	public GrassbergerMISplit(double[][] matrix){
		m_matrix = matrix;
		m_priorVal = this.entropyOverColumns();
	}
	
	public double gain(){
		return m_priorVal - this.entropyConditionedOnRows();
	}
	
	/**
	 * It estimates class entropy of the 
	 * contingency table with Grassberger 
	 * estimator. 
	 * 
	 * @return entropy value
	 */	
	public double entropyOverColumns(){
    double returnValue = 0, sumForColumn, total = 0;
    
    for (int j = 0; j < m_matrix[0].length; j++){
      sumForColumn = 0;
      for (int i = 0; i < m_matrix.length; i++)
      	sumForColumn += m_matrix[i][j];
      returnValue = returnValue - xGx(sumForColumn);
      total += sumForColumn;
    }
    if (Utils.eq(total, 0)) {
      return 0;
    }
    return returnValue/total + Math.log(total);
	}
	
	/**
	 * It estimates the conditional entropy of the columns
	 * with the Grassberger estimator.
	 * 
	 * @return conditional entropy value
	 */
  public double entropyConditionedOnRows() {
    
    double returnValue = 0, sumForRow, total = 0;

    for (int i = 0; i < m_matrix.length; i++) {
      sumForRow = 0;
      for (int j = 0; j < m_matrix[0].length; j++) {
				returnValue = returnValue + xGx(m_matrix[i][j]);
				sumForRow += m_matrix[i][j];
      }
      returnValue = returnValue - sumForRow*Math.log(sumForRow);
      total += sumForRow;
    }
    if (Utils.eq(total, 0)) {
      return 0;
    }
    return -returnValue / total;
  }

	
	/**
	 * Just a support method to compute 
	 * Entropy with the Grassberger estimator
	 * 
	 * @param x
	 * @return
	 */
	private static double xGx(double x){
		double ret = 0;
  	if (x < 1e-6)
      return ret;
  	else
  		return x*G(x);  	
	}
		
	/**
	 * It compute the closed for of G function
	 * given by Grassberger.
	 * 
	 * @param x
	 * @return
	 */
	private static double G(double x){  
	  double oddEven = 1 - 2*(x%2); // 1 if x even, -1 if x is odd
		double ret = Gamma.digamma(x) + 0.5*oddEven*(Gamma.digamma(0.5*(x+1)) - Gamma.digamma(0.5*x));
	  return ret;
  }
}
;