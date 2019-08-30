---
name: Language models, part 1
tools:
image: ../../assets/intro_nlp/her_movie_latest.gif
description:
relative_url: intro_nlp
---

## N-gram intro
<details open>
<summary markdown="span">
</summary>
A language model is a function that maps a sentence into a degree of certainty. 
Usually we normalize it to be between zero and one, so it resembles a probability but it's not.
A N-gram is the simplest language model, so let's start there.
We also call n-gram a sequence of size n:

$$
\begin{align*}
\vec{w}&=w_{1}^{n}=(w_{1},w_{2},\ldots,w_{n})\\
f(\vec{w})&=p
\end{align*}
$$

The standard way to compute a probability of a sequence of events is through the chain rule:

$$
\begin{align*}
f\left(w_{1}^{n}\right) &=f((w_{1},w_{2},\ldots,w_{N}))\\
&=f\left(w_{1}\right) f\left(w_{2} | w_{1}\right) f\left(w_{3} | w_{1}^{2}\right) \ldots f\left(w_{n} | w_{1}^{n-1}\right) \\
&=\prod_{k=1}^{n} f\left(w_{k} | w_{1}^{k-1}\right)
\end{align*}
$$

How to compute the degree of certainty of a word $w_{k}$ given a context $w_{1}^{k-1}$?
We would have to count how many times the phrase $w_{1}^{k}$ and the context $w_{1}^{k-1}$ were written.

$$
\begin{align*}
f\left(w_{k} | w_{1}^{k-1}\right)&= \frac{C(w_{1}^{k})}{C(w_{1}^{k-1})}
\end{align*}
$$
    
This is impractical since for a good estimate we would need to have a lot of data.

We can also see here that we won't be able to use standard probabilities since we cannot verify the following equality:

$$
\begin{align*}
f(w_{k-1},w_{k})&=f(w_{k-1})f(w_{k}|w_{k-1})\\
&\neq f(w_{k})f(w_{k-1}|w_{k})
\end{align*}
$$

So, we will relax the problem and come up with estimate of $f(w_{k}\|w_{1}^{k-1})$.

$$
\begin{align*}
f\left(w_{n} | w_{1}^{n-1}\right) \approx f\left(w_{n} | w_{n-1}\right)
\end{align*}
$$

Mathematicians call this approximation, Markov's assumption.
The 2-gram model then becomes:

$$
\begin{align*}
f\left(w_{1}^{n}\right) &=\prod_{k=1}^{n} f\left(w_{k} | w_{k-1}\right)
\end{align*}
$$

How do we compute $f\left(w_{k} \| w_{k-1}\right)$?

$$
\begin{align*}
f\left(w_{k} | w_{k-1}\right)&=\frac{C\left(w_{n-1} w_{n}\right)}{\sum_{w} C\left(w_{n-1} w\right)}\\
&=\frac{C(w_{k-1},w_{k})}{C(w_{k-1})}
\end{align*}
$$

Now is a good time to pause for a moment and implement this model.
```python
import numpy as np
from collections import Counter
from nltk.book import *

# text2 = sense and sensibility by jane austen
unigram_counts = Counter(text2)
bigrams = [(text2[i],text2[i+1]) for i in range(len(text2)-1)]
bigram_counts = Counter(bigrams)

def compute_bigram_probability(bigram,bigram_coutns,unigram_counts):
    return bigram_counts[bigram]/unigram_counts[bigram[0]]

def compute_sentence_bigram_probability(s,bigram_counts,unigram_counts):
    s = s.split(' ')
    return np.exp(np.sum([np.log(compute_bigram_probability((s[i],s[i+1])),bigram_counts,unigram_counts) for i in range(len(s)-1)]))

>>> compute_sentence_probability('she did not care',bigram_counts,unigram_counts)
4.569205742850947e-05
```

We can generalize these notions to n-grams instead of bigrams:

```python
def compute_ngrams(text,n):
    if len(text)!=n:
        return [tuple(text[i:(i+n)]) for i in range(len(text)-n+1)]
    else:
        return [tuple(text)]

def compute_ngram_counts(text,n):
    ngram_counts = Counter(compute_ngrams(text,n))
    context_counts = Counter(compute_ngrams(text,n-1))
    return ngram_counts,context_counts

def compute_ngram_probability(ngram,ngram_counts,context_counts,verbose=False):
    return ngram_counts[ngram]/context_counts[ngram[:len(ngram)-1]]

def compute_sentence_ngram_probability(s,ngram_counts,context_counts):
    s = s.split(' ')
    ngrams = compute_ngrams(s,len(list(ngram_counts.keys())[0]))
    return np.exp(np.sum([np.log(compute_ngram_probability(ngrams[i],ngram_counts,context_counts)) for i in range(len(ngrams))]))

>>> ngram_counts,context_counts = compute_ngram_counts(text2,n=3)
>>> compute_sentence_ngram_probability('she did not care',ngram_counts,context_counts)
0.011299435028248582
```

You can try out the code using [this jupyter notebook](https://github.com/nunoskew/language-models-part-1).
</details>

## Evaluating a Language Model
<details open>
<summary markdown='span'></summary>
We are going to use the standard machine learning model evaluation, training the model in one dataset and testing on another.
In supervised learning, we try to find a model that produces an output as close as the ground truth as possible, and in this case it will be no different.  
The direct analogy to supervised learning would be to produce a phrase and compare to what was actually written.
This would make sense if we had a mapping from descriptors to a target variable. 
Since we have a mapping of sentences to probabilities, we will compute the probability of producing the ground truth, and the we will consider the best model the one with highest probability.

For information theoretic reasons, that i hope to learn soon, instead of measuring the probability we will have a measure of deviation from the ground truth. 
It is called perplexity and it consists of the following:

$$
\begin{align*}
PP(w_{1}^{n})&=f(w_{1},w_{2},\ldots,w_{n})^{-\frac{1}{n}}\\
&=\prod\limits_{k=1}^{n}f(w_{k}|w_{k-1})^{-\frac{1}{n}}\\
&=\exp\left(\sum\limits_{k=1}^{n}\log\left(f(w_{k}|w_{k-1})^{-\frac{1}{n}}\right)\right)\\
&=\exp\left(\sum\limits_{k=1}^{n}-\frac{1}{n}\log\left(f(w_{k}|w_{k-1})\right)\right)\\
&=\exp\left(-\frac{1}{n}\sum\limits_{k=1}^{n}\log\left(f(w_{k}|w_{k-1})\right)\right)
\end{align*}
$$

We will measure the perplexity of n-grams on the same dataset we used for training, with n varying from 2 to 5.

```python
def compute_perplexity(text,ngram_counts,context_counts):
    ngrams = compute_ngrams(text,len(list(ngram_counts.keys())[0]))
    return np.exp(-(1/len(text))*np.sum([np.log(compute_ngram_probability(ngrams[i],ngram_counts,context_counts)) for i in range(len(ngrams))]))

>>> for i in range(2,6):
>>>     ngram_counts,context_counts = compute_ngram_counts(text2,i)
>>>     print('n=',i,':',compute_perplexity(text2,ngram_counts,context_counts)
n= 2 : 37.37576749369875
n= 3 : 4.6990318881662745
n= 4 : 1.5042243590107542
n= 5 : 1.095325619353706
``` 

If we try to compute perplexity or probability on new text, chances are that there are going to be some new ngrams that we did not have on our training set.This is going to make any sentence with these unseen ngrams have probability zero.
So now it's the perfect time to mention generalization. 
</details>

## Generalization
<details open>
<summary markdown="span"></summary>
> **_TODO:_** Generalization content
</details>
## Sources
* [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp)
