import os, zipfile

def create_zip(algo_name):
    base = "deployment_builds"
    zip_path = f"{base}/{algo_name}_Deploy.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        readme = f"{base}/README.txt"
        if not os.path.exists(readme):
            with open(readme, "w") as f:
                f.write("Copyright Dr Subar - MMU\n")
        z.write(readme, "README.txt")
        tfl = f"{base}/{algo_name}.tflite"
        jb = f"{base}/{algo_name}.joblib"
        if os.path.exists(tfl): z.write(tfl, os.path.basename(tfl))
        if os.path.exists(jb): z.write(jb, os.path.basename(jb))
    return zip_path
