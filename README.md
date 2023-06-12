# Identifying Causal Directions from Text - Unsupervised Learning using Bayesian Framework
Author: [King Tao Jason Ng](https://www.linkedin.com/in/jasonkng/)

## Introduction

In this project, we show empirically word occurrences resemble the characteristics of causal directions. Our aim is to gain insights into the data generation process that underlies causal directions. To achieve this, we determine a causal direction if its causal relation exists in the sentence. We propose a two-phase method: Bayesian framework and BERT, applied in two scenarios. In data augmentation, the proposed method improves the F1 score compared with BERT, but the difference is not significant. In unsupervised learning, it achieves a decent performance in contrast with random guessing. The study can be extended to capture multiple causal relations.

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