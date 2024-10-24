from contextlib import redirect_stdout
import os
root_path = "/dtu/datasets1/02516/"

dataset_folders = ["DRIVE", "PH2_Dataset_images"]
def visualize_folders(parts):
    path = os.path.join(*parts)
    subfiles = os.listdir(path)
    for file in subfiles:
        print(f"-" * len(parts) + "> " + file)
        if os.path.isdir(path + "/" + file):
            visualize_folders(parts + [file])

with open('output.txt', 'w') as f:
    with redirect_stdout(f):
        for fn in dataset_folders:
            print("\n" + "#"*(len(fn)+1) + "\n", fn, "\n" + "#"*(len(fn)+1))
            visualize_folders([root_path, fn])
