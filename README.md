# ChineseTLS

## Dataset

TLCN is our dataset. It has 77 topics and timelines. You can download the Dataset from google drive (https://drive.google.com/drive/u/1/folders/1a_Q3jAI5jqiSlBx1r8m01rjATsvPtY6Z), and you can download the raw data from here.


`datasets/TLCN` is a preprocessed TLCN dataset, `./TLCN` is the raw data.

## Library installation
To install requirements, run:
```
pip install -r requirements.txt
pip install -e .
```
[Tilse](https://github.com/smartschat/tilse) also needs to be installed for evaluation and some TLS-specific data classes.

## Run 

```
python evaluate.py  \
  --dataset datasets/TLCN/ \
  --method datewise \
  --output test15/tlcn.datewise-lk.json \
  --language chinese \
  --summarizer centriodrank \
  --date_select regression\
  --resources resources/datewise \
  --mode eval \
  --l \
  --k
```


