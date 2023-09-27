# 1. Installing the code

TODO, this has to be done manually at the moment.

# 2. Downloading the model weights and the datasets, and preprocessing the data

The segpc dataset is hosted on Kaggle, which requires an account to download and so cannot be autoamted.
So first make a free account if necessary and download the data [here](https://www.kaggle.com/datasets/sbilab/segpc2021dataset).
Leave it as a zip with the original name and put it in the datasets folder.

Now run `make` while in the main folder.

This will download and preprocess the datasets, and all the pretrained models.
It creates a directory structure:
```
datasets/
    isic2018/
        np/
    segpc2021/
        np/
saved_models/
```

Note the models take about 4.5Gb and the datasets close to 40Gb, although for the latter, everything outside of the np folders can be removed, leaving only about 4Gb in the end.

# 3. Evaluate

Evaluate all models on the test stets by running `python test_models.py datasets` (where `datasets` is the path to the directory).
Comparisons to the claimed results will be printed.    

