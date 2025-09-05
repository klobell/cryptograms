# Cryptogram Project

Generate cryptograms from quotes and predict their difficulty using machine learning.  
This project extracts features from text structure, word patterns, and cryptographic statistics to estimate difficulty.  
Motivated by the shortage of high-quality cryptograms and inconsistent difficulty ratings in existing sources.
Performance is limited by dataset size; more data should improve results.

## To run this notebook in Colab:
1. Click below to open in Google Colab:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/klobell/cryptograms/blob/main/cryptograms_github.ipynb)

2. Run:
```python
!git clone https://github.com/klobell/cryptograms.git
%cd cryptograms
```

## Data sources
Training data from [here](https://cryptograms.puzzlebaron.com/play.php)

Funny quotes from [here](https://www.rd.com/list/funniest-quotes-all-time/)

## Ethics:
- time.sleep() used to avoid overwhelming servers
- Scraping complied with all sites' robots.txt.
