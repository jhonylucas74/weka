package weka.classifiers.trees.randomForestSplitCriterionSelection;

import java.util.LinkedList;
import java.util.ListIterator;

import weka.classifiers.trees.j48.Distribution;
import weka.core.ContingencyTables;
import weka.core.Instance;
import weka.core.UnassignedClassException;
import weka.core.Utils;

/**
 * Class to compute Adjusted Mutual Information (AMI) splitting 
 * criteria. 
 * 
 * It stores a contingency table and a 3D matrix of the 
 * innermost summation probabilities. 
 *  
 * @author Simone Romano (sromano@student.unimelb.edu.au)
 *
 */  

public class AMISplit extends Split{
	  /** The natural logarithm of 2 */
    private static final double log2 = Math.log(2);
    
  	/** 
  	 * If MI is smaller that this bound 
  	 * the split is not useful. Thus, a low 
  	 * gain is returned. 
  	 */
  	private static final double epsilon_MI = 0.005;  	
  	/**
  	 * If UB/MI is smaller than this bound
  	 * no EMI is computed and the standard 
  	 * criterion is used.
  	 */
  	private static final double epsilon = 0.01;
  	
  	/**
  	 * Row marginals of the contingency table. 
  	 * They are related to the split partitions.
  	 */
  	private double[] m_a;
  	
  	/**
  	 * Column marginals of the contingency table. 
  	 * They are related to the classes.
  	 */
  	private double[] m_b;
  	
  	/**
  	 * Total elements in the contingency table.
  	 */
  	private double N;
  	
  	/**
  	 * 3D matrix to store innermost probability
  	 */
  	private double[][][] m_prob;
  	
  	/**
  	 * Class Entropy
  	 */
  	protected double m_priorVal;
  	
  	/** Constructor. It uses a 3-dimensional matrix to store innermost summation
  	 * probabilities. That matrix is also used in the 
  	 * incremental version of AMI computation. The  	 
  	 * incremental version is used only in continuous attributes. 
  	 * 
  	 * If the probability matrix is initialized with a given contingency table then
  	 * it can be used straight away to compute AMI. Otherwise, it 
  	 * can be used as starting point for incremental AMI computation.
  	 * 
  	 * @param matrix contingency matrix
  	 */
  	
  	public AMISplit(double[][] matrix){
  		m_matrix = matrix;
  		  		
  		// init some marginals
      int i, j;

      m_a = new double [matrix.length];
      m_b = new double [matrix[0].length];
      for (i = 0; i < matrix.length; i++) 
        for (j  = 0; j < matrix[i].length; j++) {
			  	m_a[i] += matrix[i][j];
			  	m_b[j] += matrix[i][j];
			  	N += matrix[i][j];
        }
      /*
      System.out.println(N);
      for(i=0; i < m_matrix.length; i++){
  		  for (j = 0; j < m_matrix[0].length; j++) {
  		    System.out.print(m_matrix[i][j] + " ");
  		  }
  		  System.out.println();
      }
  	  System.out.println();
      */
  		// init class entropy
  		m_priorVal = ContingencyTables.entropyOverColumns(m_matrix);
  		// init 3d matrix
  		this.initProbMatrix();
  	}
  	
  	/**
  	 * Overrides gain methods in order to 
  	 * provide AMI as splitting criterion.
  	 */
  	public double gain(){
  		/*
  	  System.out.println();
      for(int i=0; i < m_matrix.length; i++){
  		  for (int j = 0; j < m_matrix[0].length; j++) {
  		    System.out.print(m_matrix[i][j] + " ");
  		  }
  		  System.out.println();
      }
      */
  		return this.AMI();
  	}
  	
  	/**
  	 * Initialize the probability matrix regarding 
  	 * the current contingency table. AMI can be computed 
  	 * straight away then.
  	 */  	
  	private void initProbMatrix(){
  		/* Initialize a 3D matrix that is also used in
  		 * the incremental version of AMI. The third dimension
  		 * should big as bis as the maximum value across
  		 * all marginals.  
  		 */
  		double maxClass = m_b[Utils.maxIndex(m_b)];
  		double maxSplitSet = m_a[Utils.maxIndex(m_a)];
  		int thirdDim = (int)Math.max(maxClass, maxSplitSet);
  		m_prob = new double[m_a.length][m_b.length][thirdDim];
  		
    	double nij; // used in the innermost for loop
    	double inf, sup; // auxiliary variables
    	double p0;
  		for (int i=0; i < m_a.length; ++i){
  			for (int j=0; j < m_b.length; ++j){ 
  				inf = Math.max(1, m_a[i] + m_b[j] - N);
  				sup = Math.min(m_a[i],m_b[j]);
  				nij = inf;
  				p0 = p(nij,m_a[i],m_b[j],N);
  				m_prob[i][j][(int)(nij-1)] = p0;		
  				for ( nij = inf + 1; nij <= sup; ++nij ){
  					p0 *= (m_a[i] - nij + 1)*(m_b[j] - nij + 1)/nij/(N-m_a[i]-m_b[j]+nij);
  					m_prob[i][j][(int)(nij-1)] = p0;
  				}
  			}
  		}
  	}
  	
  	
  	/**
  	 * Method to compute the innermost summation probability:
  	 * 
  	 * ai!bj!(N-ai)!(N-bj)! 
  	 * ____________________
  	 * 
  	 * N!nij!(ai-nij)!(bj-nij)!(N-ai-bj+nij)!
  	 * 
  	 * @param nij (as in the formula).
  	 * @param ai (as in the formula).
  	 * @param bj (as in the formula).
  	 * @param N (as in the formula).
  	 * @return 
  	 */
  	private double p(double nij, double ai, double bj, double N){
    	double x_1, x_2, v;
    	double p;
    	
  		// Two linked lists in order to avoid precision errors
  		LinkedList<Double> num = new LinkedList<Double>();
  		LinkedList<Double> den = new LinkedList<Double>();
  		
  		// Some simplifications on numerator and denominator
  		x_1 = Math.min(nij, N - ai - bj + nij); 
  		x_2 = Math.max(nij, N - ai - bj + nij);
  		// numerator
  		v = ai;
  		while (v > ai - nij)
  			num.add(v--);
  		v = bj;
  		while (v > bj - nij)
  			num.add(v--);
  		// denominator
  		v = N;
  		while (v > N - ai)	
  			den.add(v--);
  		v = x_1;
  		while (v > 0)
  			den.add(v--);
  		// Due to the simplifications
  		if (N - bj > x_2)	{ // add a term to numerator
  			v = N - bj;
  			while (v > x_2)
  				num.add(v--);
  		} else {            // add a term to denominator
  			v = x_2;
  			while (v > N - bj)
  				den.add(v--);
  		}
  		// num and den have the same number of terms
  		p = 1;
  		ListIterator<Double> denIt = den.listIterator();
  		for (double x : num)
  			p *= x/denIt.next();
  		return p;
    }
  	
  	/**
  	 * Computes the AMI for the given contingency table.
  	 * 
  	 */
  	private double AMI(){
  		double MI, UB;
  		
  		MI = m_priorVal - ContingencyTables.entropyConditionedOnRows(m_matrix); 
  		UB = this.AMIupperBound();
  		//System.out.println("MI " + MI);
  		//System.out.println("UB " + UB);
  		/*
  		 * If MI is very small
  		 * discard this split
  		 */
  		if (MI <= epsilon_MI){
  			//System.out.println("Small MI");
  			return Double.NEGATIVE_INFINITY;
  		}
  		/*
  		 * If UB/MI is very small
  		 * it is not necessary to
  		 * compute EMI 
  		 */
  		if (UB <= epsilon*MI){
  			
  			//System.out.println("Small ub");
  			return MI/m_priorVal;
  		}
  		//Finally compute EMI
  		double EMI = 0;
  		double inf, sup;
	  	for (int i=0; i < m_prob.length; i++){
	  		for (int j=0; j < m_prob[i].length; j++){
	  			inf = Math.max(1, m_a[i] + m_b[j] - N);
	  			sup = Math.min(m_a[i],m_b[j]);
	  			for ( double nij = inf; nij <= sup; ++nij ){
	  				//System.out.println(m_prob[i][j][(int)(nij-1)]);
	  				EMI += nij * Math.log(nij/N) * m_prob[i][j][(int)(nij-1)];
	  			}
	        if ( m_a[i] > 0 && m_b[j] > 0)
	        	EMI -= m_a[i]*m_b[j]/N * Math.log(m_a[i]*m_b[j]/N/N);
	      }
	  	}
	  	EMI /= N*log2;
  		
  		/*
  		 * Error condition: |EMI| >= UB
  		 * it might happen the value 1 
  		 * occurs exactly once in each row 
  		 * or column
  		 */
  		if (Math.abs(EMI) >= UB){
  			//System.out.println("Error in EMI = " + EMI);
  			return MI/m_priorVal;
  		}
  		
  		double AMI = (MI - EMI)/(m_priorVal - EMI);
  		
  		//System.out.println(EMI);
  		//System.out.println(AMI);
  		
  		return AMI;
  	}
  	

  	/**
  	 * It updates the probabilities values for the 3D matrix
  	 * given in input. The contingency table given in input should
  	 * differ by only 1 cell from the one used to initialize the 
  	 * 3D matrix 
  	 * 
  	 * @param from The bag to shift 1 instance from
  	 * @param to The bag to shift 1 instance to
  	 */
  	private void incrementProbMatrix(int from, int to){
  		double inf, sup, nij;
  		// cells related to the row "from"
  		for( int j = 0; j < m_b.length; j++){
  			inf = Math.max(1, m_a[from] + m_b[j] - N);
  			sup = Math.min(m_a[from],m_b[j]);
  			// different summation boundaries than the previous one
  			if (m_a[from] + m_b[j] - N >= 1){
  				m_prob[from][j][(int)(inf-1)] = m_prob[from][j][(int)(inf)]*(m_a[from]+m_b[j]-N+1)/(m_a[from]+1);//p(ai+bj-N,ai,bj,N);
  				inf++;
  			}
  			for ( nij = inf; nij <= sup; ++nij )
  				m_prob[from][j][(int)(nij-1)] = m_prob[from][j][(int)(nij-1)]*(N-m_a[from])*(m_a[from]+1-nij)/(m_a[from]+1)/(N-m_a[from]-m_b[j]+nij);  		
  		}
  		// cells related to the row "to"
  		for( int j = 0; j < m_b.length; j++){
  			inf = Math.max(1, m_a[to] + m_b[j] - N);
  			sup = Math.min(m_a[to],m_b[j]);
  			// different summation boundaries than the previous one
  			if (m_a[to] <= m_b[j]){
  				m_prob[to][j][(int)(sup-1)] = m_prob[to][j][(int)(sup-2)]*(m_b[j]-m_a[to]+1)/(N-m_a[to]+1);//p(ai,ai,bj,N);
  				sup--;
  			}
  			for ( nij = inf; nij <= sup; ++nij )
  				m_prob[to][j][(int)(nij-1)] = m_prob[to][j][(int)(nij-1)]*m_a[to]*(N-m_a[to]+1-m_b[j]+nij)/(m_a[to]-nij)/(N-m_a[to]+1);  		
  		}
  	}
  	
  	/**
  	 * Every time the shift method is called the probability 
  	 * matrix has to be updated.
  	 */
  	public void shift(int from, int to, Instance instance) throws UnassignedClassException{
  		super.shift(from, to, instance);
  		// update marginals
  		m_a[from]--;
  		m_a[to]++;
  		// increment 3d matrix
  		this.incrementProbMatrix(from, to);
  	}
  	
  	/**
  	 * It computes the tighter upper bound on EMI
  	 * for a given contingency table. 
  	 * 
  	 * @param bags as j48 Distribution
  	 * @return upper bound on EMI
  	 */
  	private double AMIupperBound(){
  		double UB = 0;
  		for (int i=0; i < m_a.length ; ++i)
  			for (int j=0; j < m_b.length; ++j){
          if ( m_a[i] > 0 && m_b[j] > 0)
          	UB += m_a[i]*m_b[j]*Math.log( N * ((m_a[i]-1)*(m_b[j]-1) + N-1)
          														/ (m_a[i]*m_b[j]*(N-1)) );
  			}
  		return UB/N/N/log2;  		
  	}
  	
}
