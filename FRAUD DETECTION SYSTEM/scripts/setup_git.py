import os
import sys
import urllib.request
import zipfile

def install_mingit():
    url = "https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.1/MinGit-2.47.1-64-bit.zip"
    zip_path = os.path.abspath("mingit.zip")
    dest_dir = os.path.abspath(".git-bin")

    print(f"Downloading MinGit from {url}...")
    urllib.request.urlretrieve(url, zip_path)
    print("Download complete. Extracting files...")
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
    if os.path.exists(zip_path):
        os.remove(zip_path)
    print(f"Git installed successfully to: {dest_dir}")

if __name__ == "__main__":
    install_mingit()
