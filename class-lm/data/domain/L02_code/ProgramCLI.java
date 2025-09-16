public class ProgramCLI {
	
	public static void main (String[] args) {
		int sum = 0;

		if(args== null)
			System.out.println("Args is null");
		
		if(args != null)
			for(int i = 0;  i< args.length; i++)
				sum = sum + Integer.valueOf(args[i]);
		
		System.out.println(sum);
	}
	
}