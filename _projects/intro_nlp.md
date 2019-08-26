---
name: Language models, part 1
tools:
image: ../../assets/intro_nlp/her_movie_latest.gif
description:
relative_url: intro_nlp
---

A language model is a function that maps a sentence into a degree of certainty. 
Usually we normalize it to be between zero and one, so it resembles a probability but it's not.
A N-gram is the simplest language model, so let's start there.
We also call N-gram a sequence of size N:

$$
\begin{align*}
\vec{w}&=(w_{1},w_{2},\ldots,w_{N})\\
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

So, we will relax the problem and in the process we'll come up with estimate of $f(w_{k}\|w_{1}^{k-1})$.

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
> **_TODO:_** Implement bigram 
## Sources
* [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp)
