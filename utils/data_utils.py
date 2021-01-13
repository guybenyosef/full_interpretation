from pathlib import Path

def make_dataset(dir, ext='jpg'):
    nparrays = []
    for fname in Path(dir).glob('**/*.{}'.format(ext)):
        nparrays.append(str(fname))

    return nparrays

def make_dataset_txtfile(filename):
    f = open(filename, "r")
    #nparrays = f.readlines()
    nparrays = f.read().splitlines()
    f.close()

    return nparrays

