//Same as in Code0, to write a functon that cann do addition, I need to write a class...
//Since those functions does not really need an object to make sense, they are static.
//You will notice that I am using the same name for the 2 functions (using Overloading)
//NB- Those comments are example of bad comments... You will see better comments in Program4.java
public class Program1 {
	
	public static int addition(int a, int b) {
		return a+b;
	}
	
	public static String addition(String a, String b) {
		return a+b;
	}
	
	public static void main(String[] args) {
		int result = addition(5, 6);
		System.out.println(result);
		
		String result2 = addition("Welcome", " CS310ers");
		System.out.println(result2);
	}
}