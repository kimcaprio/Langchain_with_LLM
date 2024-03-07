import subprocess

print(subprocess.run(["sh LangChain/ModelDownload/download_models.sh"], shell=True))