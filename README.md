## Authorship Attribution
###### Models to predict authors and book based on lexical features.

Use the following command to execute the program.  
`python run.py`

Files:
- [run.py](run.py): runs the models and run evaluation.
- [tools/models.py](tools/models.py): contains the author attribution and language model classes.
- [tools/util.py](tools/util.py): contains some useful utilities.
- [data/raw](data/raw): raw data is stored here
- [data](data): after processing, the processed data is stored here
- [models/lm](models/lm): language models for each of the authors is stored here
- [models/clf](data/raw): classifier models are stored here

#### Code
The following modules are required to run the system:

  * Python 2.7
  * NumPy
  * Pandas
  * Scipy
  * scikit-learn
  * nltk
  * cPickle v1.71
