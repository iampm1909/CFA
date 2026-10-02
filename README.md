# CFA
For reproducing the results, run the following commands after downloading the requirements text file and both the python codes onto google colab;
   !pip install -q -r requirements_basic.txt
   !python -m spacy download en_core_web_sm
   !python fetch_first5.py
   !python master_basic.py

   All the csv files mentioned are outputs with documents and sentences laying out all the documents and sentences that are being fetched alongside giving indexes, while the statements, edges and diagnostics files help with the criteria matching and a generic picture to be formed out of the master script as was kept in mind.

   Please note: As this was a trial, the first 5 entries from the grey literature document were hardcoded to be fetched so as to make the job a bit easier and can be scaled accordingly to fetch without hardcoding once there has been a decision on the master script.
