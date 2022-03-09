from shutil import copyfile


src = "../cmake-build-debug/"
dst = "../Python/"

files = [
    "Nano.py",
    "_Nano.so",
]

for file in files:
    copyfile(src + file, dst + file)
