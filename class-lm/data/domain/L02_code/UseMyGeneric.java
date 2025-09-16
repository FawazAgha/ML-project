class Car{}
class House{}

public class UseMyGeneric { //using the generic class

    public static void main(String[] args) {
        MyGeneric myGenericIns = new MyGeneric();
        Car c1 = new Car();
        
        myGenericIns.store(c1);
        Car c2 = (Car) myGenericIns.getStoredObject();
    }
}
