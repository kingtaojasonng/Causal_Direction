# Identifying Causal Directions from Text - Unsupervised Learning using Bayesian Framework
Author: [King Tao Jason Ng](https://www.linkedin.com/in/jasonkng/)

## Introduction

Our project is to determine a causal direction if its causal relation exists in a sentence, especially when no training data is provided. We want to find out whether the Bayesian framework boosts performance for Bidirectional Encoder Representations from Transformers (BERT). Hence, we propose a two-phase method: 1. Bayesian framework, which augments data by incorporating word occurrences from the Internet's domains. 2. BERT, which utilises semantics of words based on the context to perform classification. In data augmentation, the proposed method boosts F1 score ever slightly compared with BERT alone, but the difference is not statistically significant. In unsupervised learning, the proposed method does achieve a decent performance in contrast to random guessing. We show empirically word occurrences resemble the characteristics of causal directions to the extent of certainty that the Bayesian framework can qualify.

## Folder Structure

<pre>
.
├── data
│   ├── SemEval2007_task4
│   └── SemEval2010_task8
├── implementation
│   ├── augmentation.py
│   ├── bayesianinference.py
│   ├── bert.py
│   ├── datapreprocessing.py
│   ├── experiment.py
│   ├── lime.ipynb
│   └── prior.py
└── README.md
    
</pre>

## Contributions

Thank [Diego Molla-Aliod](https://researchers.mq.edu.au/en/persons/diego-molla-aliod), [Rolf Schwitter](https://researchers.mq.edu.au/en/persons/rolf-schwitter) and [Houying Zhu](https://researchers.mq.edu.au/en/persons/houying-zhu) for their advice and supervision.