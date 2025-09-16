


public class MyMarketPlace {

    Sellable product;

    public MyMarketPlace(Sellable product) {
        this.product = product;
    }

    public boolean sell(double offer) {
        if(product.getPrice() > offer)
            return true;
        else
            return false;

    }
}