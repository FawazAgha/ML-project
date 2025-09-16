//See an example of Class that "use" a Generic class Operation
//Please read the file Operation.java before
public class Program2 {
	
	
	public static void main(String[] args) {
		
		Operation<Integer> op1 = new Operation<>();
		int result = op1.returnMax(5, 6);
		System.out.println(result);

		int a = 10;
		int b = 0;
		
		
		Operation<String> op2 = new Operation<>();
		String result2 = op2.returnMax("Welcome", " CS310ers");
		System.out.println(result2);

		System.out.println(a/b);
	}
}