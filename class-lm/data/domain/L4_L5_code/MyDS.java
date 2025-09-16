//Toy example for a Static Array. 
//With Static Array: Capacity will not grow to fit new data.

public class MyDS<T> implements DSInterface<T> , Iterable<T> {
    //Underlying Structure
    private T[] elements; //storage
    
    private int size = 0; //Optional but highly recommended
    private int capacity = 1; //Optional but highly recommended

    @SuppressWarnings("unchecked")
    public MyDS() {
        // As discussed in class, one cannot do: elements = new T[capacity];
        elements = (T[]) new Object[this.capacity]; //Notice the downcasting.
    }

    @SuppressWarnings("unchecked")
    public MyDS(int cap) {
        //Public Method ? --- Maybe some validation of the arguments?
        elements = (T[]) new Object[cap];
        this.capacity = cap;
		//elements.length
    }

    public T get(int index) {
		if(index<0)
			throw new Exception();
		//[A,B,C,-,-,-]
		if(index>=size)
			throw new Exception();

			
		//[A,B,C,null,null,null] size=3
        //Public Method ? --- Maybe some validation of the arguments?
        return elements[index];
    }

    public T set(int index, T elt) {
        //Public Method ? --- Maybe some validation of the arguments?
        T old = elements[index];
        elements[index] = elt;
        return old;
    }

    public boolean append(T elt) {
		//O(1) amortized analysis

        if(size<capacity) {
            elements[size] = elt;
			size++;
            return true;
        }
        else {
            //Do not adjust the capacity, it is a Static Array
            
			return false;
        }
    }
	
	int size() //O(1). 

    public T remove() {
        T old = elements[--size];
        return old;
    }

    public boolean insert(int index, int elt) {
        //Your turn: Implement this just for extra practice.
        throw new UnsupportedOperationException();
    }

    public T delete(int index) {
        //Your turn: Implement this just for extra practice.
        throw new UnsupportedOperationException();
    }

    public T search() {
        //Your turn: Implement this just for extra practice.
        throw new UnsupportedOperationException();
    }

    //Additional methods
    public void printElements() {
        //Easy to access and display the data. We are inside of the DS class itself.
        for(int i=0; i<size; i++) {
            System.out.println(elements[i]);
        }
    }

    
    //Using Inner Class. Using anonymous class is slightly different
    class InnerIterator implements Iterator<T> {

        int current = 0;

        @Override
        public boolean hasNext() {
            if(current < size)
                return true;
            return false;
        }

        @Override
        public T next() {
			T elt = elements[current];
			current++;
            return elt;
        }

    }

    //Using anonymous class...
    @Override
    public Iterator<T> iterator() {
				
        return new Iterator<>() {
			int current = 0;

			@Override
			public boolean hasNext() {
				if(current < size)
					return true;
				return false;
			}

			@Override
			public T next() {
				if(hasNext())
					return elements[current++];
				throw new java.util.NoSuchElementException();
			}
		};
    }

    
    
}