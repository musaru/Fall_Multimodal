import os
import shutil

for i in range(12,18):
    for filename in os.listdir(f"./HAR_UP/Subject{i}"):
        if filename.endswith(".zip"):
            print(os.path.join(f"./HAR_UP/Subject{i}",filename))
            shutil.unpack_archive(os.path.join(f"./HAR_UP/Subject{i}",filename),os.path.join(f"./HAR_UP/Subject{i}"))