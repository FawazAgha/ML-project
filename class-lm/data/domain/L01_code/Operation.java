//This Generic class has a method (not quite the same as function) that can add two of "almost" any Object...
//Notice that we used <E extends Comparable<E>> instead of <E>
//This is because we do not programmer to use it with just any objects... we add a restriction: This object must implement Comparable...
//We add this restriction because otherwise, we would not have the guarantee that we can have a method compareTo to use in line 8.\
//E can be replaced later on (for instance in Program2.java) with any classes that implements Comparable.

public class Operation<AnyType extends Comparable<AnyType>> {
	
	public AnyType returnMax(AnyType a, AnyType b) {
		if(a.compareTo(b)<0)
			return b;
		return a;
	}

}