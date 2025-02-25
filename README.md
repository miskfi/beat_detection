# Beat detection and tempo analysis of musical recordings (NI-SCR semestral project)

## About

This project builds a simple deep learning model that analyzes audio recordings of ballroom dance music (such as waltz, tango, jive, cha-cha and many more) and tries to detect where (or when) the beats occur in those recordings. That information can then be used to infer the tempo of the song, measured in beats per minute (BPM).



The `report.ipynb` Jupyter notebook contains all information about the dataset, model, training, evaluation, visualization & discussion. Most of the custom code can be found in the respective Python source files.



## Get started

### Install packages

To install the necessary packages, run:

```bash
pip install -r requirements.txt
```



### Download the data

The dataset is be downloaded automatically via the `mirdata` package (see the Jupyter notebook).