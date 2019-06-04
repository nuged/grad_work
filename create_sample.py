import os, sys, shutil

try:
    N_SAMPLES = int(sys.argv[2])
    mode = sys.argv[1]
except IndexError:
    print 'Enter the number of samples AND mode'
    exit(1)

if mode.lower().startswith('tr'):
    source = 'training'
    dest = 'small_training'
elif mode.lower().startswith('te'):
    source = 'testing'
    dest = 'small_testing'
else:
    raise AttributeError('Wrong mode')

os.chdir('mnist')

try:
    shutil.rmtree(dest)
except OSError:
    pass

try:
    os.mkdir(dest)
except OSError:
    pass

for i in range(10):
    dest_path = os.path.join(dest, str(i))
    source_path = os.path.join(source, str(i))
    try:
        os.mkdir(dest_path)
    except OSError:
        os.remove(dest_path)
    for file in os.listdir(source_path):
        if int(file[:-4]) < N_SAMPLES // 10:
            shutil.copy2(os.path.join(source_path, file), dest_path)
