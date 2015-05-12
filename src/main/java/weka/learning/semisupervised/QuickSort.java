package weka.learning.semisupervised;

public class QuickSort {
    
    private double array[][];
    private int length;
 
    public void sort(double[][] inputArr,int length) {
         
        if (inputArr == null || inputArr.length == 0) {
            return;
        }
        this.array = inputArr;
        this.length = length;        
        quickSort(0, length - 1);        
    }
 
    private void quickSort(int lowerIndex, int higherIndex) {
         
        int i = lowerIndex;
        int j = higherIndex;
        // calculate pivot number, I am taking pivot as middle index number
        double pivot = array[1][lowerIndex+(higherIndex-lowerIndex)/2];
        // Divide into two arrays
        while (i <= j) {
            /**
             * In each iteration, we will identify a number from left side which
             * is greater then the pivot value, and also we will identify a number
             * from right side which is less then the pivot value. Once the search
             * is done, then we exchange both numbers.
             */
            while (array[1][i] < pivot) {
                i++;
            }
            while (array[1][j] > pivot) {
                j--;
            }
            if (i <= j) {
                exchangeNumbers(i, j);
                //move index to next position on both sides
                i++;
                j--;
            }
        }
        // call quickSort() method recursively
        if (lowerIndex < j)
            quickSort(lowerIndex, j);
        if (i < higherIndex)
            quickSort(i, higherIndex);
    }
 
    private void exchangeNumbers(int i, int j) {
    	double temp_1 = array[1][i];
    	double temp_0 = array[0][i];
        array[1][i] = array[1][j];
        array[0][i] = array[0][j];
        array[1][j] = temp_1;
        array[0][j] = temp_0;
    }
    
    public void printArray(double array[][],int length){
    	if (array == null){
    		return;
    	}
    	
    	for( int i=0; i<length; i++){
    		System.out.println("> "+array[0][i]+" - "+array[1][i]);
    	}
    	
    }
     
    public static void main(String a[]){
         
    	QuickSort sorter = new QuickSort();
        double[][] input = {{ 1,2,3,4,5,6,7,8,9,10,11 },{24,2,45,20,56,75,2,56,99,53,12 }};
        sorter.sort(input,11);
        sorter.printArray(input,11);
        
    }
}
