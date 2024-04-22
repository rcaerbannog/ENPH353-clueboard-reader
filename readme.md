Repo for letter classification model (CNN architecture) training + testing.

Training data generator is `text-generator-solid-v2.ipynb`. This should make 36 subfolders inside the `train` folder (one for each character of `A-Z` and `0-9`), each containing 16 training images, for a total of 576. These images feature 4 different font sizes, with/without distortion, and with/without erosion. 

CNN Trainer is located at `train/train-cnn.ipynb`. Run the script to generate a model file `letter-recog-model.keras` (or `letter-recog-model.h5`) inside the `train` directory and copy it to your competition controller package (https://github.com/rcaerbannog/enph353-HumanAdjacent) inside the `src` folder if not already present. (The `clueboards.py` script over there imports the model file.) Make sure you run the training script with the SAME or an EARLIER version of Tensorflow and Keras than the machine you want to run the compeition on, as the model files are NOT guaranteed to be backwards-compatible! (Versions 2.9.3, 2.13.1, and 2.16.x are confirmed to not be backwards-compatible.)

The model used during our competition run is located inside the `best-model-tf2.9.3-num-all` folder (indicating that it was trained on Tensorflow 2.9.3, with the number `0-9` included in the training data as well as letters, and with the entire dataset instead of with a validation split).

The initial experimentation notebooks and development history up to Sunday Apr 7 are located at https://github.com/rcaerbannog/ENPH353_Local. (Let me know if unable to access.)
