package weka.learning.semisupervised;

import weka.core.Instance;

public class InstanceMeta {
	
	private int intanceIndex;
	private double classValue;						// classe da instancia i
	private int clusterValue;						// grupo da instancia i
	private double maxClassValue;
	
	public InstanceMeta () {
		
	}
	
	public InstanceMeta (int instanceIndex, double classValue, int clusterValue, double[] distClassValue) {
		this.setIntanceIndex(instanceIndex);
		this.setClassValue(classValue);
		this.setClusterValue(clusterValue);
		this.setMaxClassValue(distClassValue);
	}
	
	/**
	 * @return the intanceIndex
	 */
	public int getIntanceIndex() {
		return intanceIndex;
	}
	/**
	 * @param intanceIndex the intanceIndex to set
	 */
	public void setIntanceIndex(int intanceIndex) {
		this.intanceIndex = intanceIndex;
	}
	/**
	 * @return the classValue
	 */
	public double getClassValue() {
		return classValue;
	}
	/**
	 * @param classValue the classValue to set
	 */
	public void setClassValue(double classValue) {
		this.classValue = classValue;
	}
	/**
	 * @return the clusterValue
	 */
	public int getClusterValue() {
		return clusterValue;
	}
	/**
	 * @param clusterValue the clusterValue to set
	 */
	public void setClusterValue(int clusterValue) {
		this.clusterValue = clusterValue;
	}
	
	public void setMaxClassValue(double[] a) {
		double max = a[0];
	    for(int i = 1; i < a.length; i++){
	      if(a[i] > max)
	    	  max = a[i];
	    }
		this.maxClassValue = max;
	}
	
	/**
	 * @return maxClassValue
	 */
	public double getMaxClassValue() {
		return this.maxClassValue;
	}
}
