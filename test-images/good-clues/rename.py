import os

folders = ['crime-scare', 'size-three', 'time-winter', 'victim-kangaroo']

for folder in folders:
    files = os.listdir(folder)
    for i in range(len(files)):
        os.rename(os.path.join(folder, files[i]), os.path.join(folder, f'testimg-{i:>03}.jpg'))