# Overview

This repository contains the experimental results, data analysis, and model evaluation code from "Evaluating vector-space models of analogy" (Chen, Peterson, & Griffiths, to appear in the *CogSci 2017 Proceedings*). You can download the paper [here](http://www.dawnchen.info/papers/vector_space_analogy_CogSci_2017.pdf).

Each folder corresponds to one of the experiments. `relsim` contains the results, analysis, and model evaluation for Experiment 1 (basic relational similarity). `symmetry` contains everything pertaining to Experiment 2 (violations of symmetry in relational similarity judgments). `transitivity` corresponds to Experiment 3 (violations of transitivity in analogy quality judgments).


# Datasets

There are two versions of each experiment's dataset. The "raw data" version has individual ratings and includes participants who were excluded from data analysis due to failing attention checks. (For Exp 1, the left-right order of the word pairs is shown.) The "mean ratings" version just has the mean ratings with those participants excluded. (For Exp 1, the mean is taken over both presentation orders.)


# Regarding Word Vectors

The model evaluation code requires a `.pickle` file containing the word-vector dictionary. We have not included any dictionaries due to their size, but they can be created by downloading the pre-trained [word2vec](https://code.google.com/archive/p/word2vec/) or [GloVe](https://nlp.stanford.edu/projects/glove/) vectors (for GloVe, we used the Common Crawl with 840B tokens version), then running `bin2text.c` for the word2vec vectors only, and finally running `store_word_dict.py`. Of course, other word-vector dictionaries may also be used. In particular, if running the model evaluation code on a system with limited memory, it may be desirable to use only a subset of the word vectors that are actually needed.


# Dependencies/Credits

The code depends on [NumPy](http://www.numpy.org/), [SciPy](https://www.scipy.org/), [pandas](http://pandas.pydata.org/), and [StatsModels](http://www.statsmodels.org/stable/index.html). `bin2text.c` is a modification of `distance.c` from Google's word2vec code.
