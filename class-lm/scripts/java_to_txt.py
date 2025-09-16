# train/rename_java_to_txt.py
import os, sys

def rename_java_to_txt(folder):
    for filename in os.listdir(folder):
        if filename.endswith(".java"):
            base = os.path.splitext(filename)[0]
            old_path = os.path.join(folder, filename)
            new_path = os.path.join(folder, base + ".txt")
            os.rename(old_path, new_path)
            print(f"Renamed {filename} -> {base}.txt")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rename_java_to_txt.py <folder>")
        sys.exit(1)
    rename_java_to_txt(sys.argv[1])
