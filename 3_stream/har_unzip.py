import os
import shutil

for i in range(12,18):
    for root,dirs,files in os.walk(f"./HAR_UP/Subject{i}"):
        for j,file in enumerate(sorted(files)):
            if file.endswith(".zip") and file.startswith("Subject"):
                print(os.path.join(root,file))
                if os.path.exists(os.path.join(root,"camera0")):
                    shutil.rmtree(os.path.join(root,"camera0"))
                shutil.unpack_archive(os.path.join(root,file),os.path.join(root,f"camera{j+1}"))