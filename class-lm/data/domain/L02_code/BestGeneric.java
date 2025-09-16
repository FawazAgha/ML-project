public class BestGeneric<AnyType> {
    
    public AnyType stored;

    public void store(AnyType toStore) {
        stored = toStore;
    }

    public AnyType getStoredObject(){
        return stored;
    }
}


//public class BestGeneric<AnyType extends Sellable> {

