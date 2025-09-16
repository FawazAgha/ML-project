public interface DSInterface<T> {
    public T get(int index);
    public T set(int index, T elt);
    public boolean append(T elt);
    public T remove();
    public boolean insert(int index, T elt);
    public T delete(int index);
    public T search();
}
