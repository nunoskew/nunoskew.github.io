---
name: Introduction to Machine Learning 
tools:
image: ../../assets/intro_ml/chris_bishop_ml_book.jpg
description:
relative_url: intro_ml 
---

## The Model

$$
\begin{align*}
y(\vec{w})=w_{0}+w_{1}x
\end{align*}
$$

## The Error Function

$$
\begin{align*}
E[\vec{w}]=\frac{1}{2}\sum\limits_{i=1}^{m}(y(x_{i},\vec{w})-t_{i})^{2}
\end{align*}
$$

## Minimizing the Error Function

We want to find the $\underset{\vec{w}}{\operatorname{argmin}} E[\vec{w}]$ 
Lets start by setting up the notation:

Let $r_{i}=y_{i}(\vec{w})-t_{i}=w_0+w_{1}x_{i}-t_{i}$. So $E[r_{i}] =\frac{r_{i}^{2}}{2}$ and $E[\vec{w}]=\frac{1}{2}\sum\limits_{i=1}^{m}r_{i}^{2}$

$$
\begin{align*}
\frac{dE}{dr_{i}}&=\frac{1}{2}*2r_{i}=r_{i}\\
\frac{dr_{i}}{dw_{0}}&=1\\
\frac{dr_{i}}{dw_{1}}&=x_{i}\\
\frac{dE}{dw_{0}}&=\sum\limits_{i=1}^{m} \frac{dE}{dr_{i}}\frac{dr_{i}}{dw_{0}}=\sum\limits_{i=1}^{m}\frac{dE}{dr_{i}}=\sum\limits_{i=1}^{m}r_{i}=\sum\limits_{i=1}^{m}w_0+w_{1}x_{i}-t_{i}\\
\frac{dE}{dw_{1}}&=\sum\limits_{i=1}^{m} \frac{dE}{dr_{i}}\frac{dr_{i}}{dw_{1}}=\sum\limits_{i=1}^{m}\left(w_0+w_{1}x_{i}-t_{i}\right)x_{i}\\
\end{align*}
$$

Ok, all set!

$$
\begin{align*}
\frac{dE}{dw_{0}}=0&\leftrightarrow \sum\limits_{i=1}^{m}w_0+w_{1}x_{i}-t_{i}=0\\
&\leftrightarrow mw_0+\sum\limits_{i=1}^{m}w_{1}x_{i}-t_{i}=0\\
&\leftrightarrow m w_{0}=\sum\limits_{i=1}^{m} t_{i}-w_{1}x_{i}\\
&\leftrightarrow m w_{0}=\sum\limits_{i=1}^{m} t_{i}-\sum\limits_{i=1}^{m} w_{1}x_{i}\\
&\leftrightarrow m w_{0}=\sum\limits_{i=1}^{m} t_{i}-w_{1}\sum\limits_{i=1}^{m} x_{i}
&\rightarrow w_{0}=\frac{1}{m}\left (\sum\limits_{i=1}^{m} t_{i}-w_{1}\sum\limits_{i=1}^{m} x_{i}\right)
\end{align*}
$$

Defining:

$$
\begin{align*}
\overline{x}&=\sum\limits_{i=1}^{m}x_{i}\\
\overline{t}&=\sum\limits_{i=1}^{m}t_{i}\\
\end{align*}
$$

The first model parameter $w_{0}$ becomes:

$$
\begin{align*}
w_{0}=\frac{1}{m}(\overline{t}-w_{1}\overline{x})
\end{align*}
$$

Applying the same procedure to $\frac{dE}{dw_{1}}$:

$$
\begin{align*}
\frac{dE}{dw_{1}}=0&\leftrightarrow\sum\limits_{i=1}^{m}\left(w_0+w_{1}x_{i}-t_{i}\right)x_{i}=0\\
&\leftrightarrow \sum\limits_{i=1}^{m} w_{0}x_{i}+w_{1}x_{i}^{2}-x_{i}t_{i}=0\\
&\leftrightarrow \sum\limits_{i=1}^{m} w_{1}x_{i}^{2}=\sum\limits_{i=1}^{m} x_{i}t_{i}-w_{0}x_{i}\\
\end{align*}
$$

Defining:

$$
\begin{align*}
\overline{x^{2}}&=\sum\limits_{i=1}^{m} x_{i}^{2}\\
x^{T}t&=\sum\limits_{i=1}^{m} x_{i}t_{i}
\end{align*}
$$

The second model parameter $w_{1}$ that minimizes the mean squared error then satisfies the following:

$$
\begin{align*}
w_{1}\overline{x^{2}}&=x^{T}t-w_{0}\overline{x}\\
\end{align*}
$$

So we now want to find $w_{0}$ e $w_{1}$ that satisfy the following equations: 

$$
\begin{align*}
w_{0}&=\frac{1}{m}\left (\overline{t}-w_{1}\overline{x}\right)\\
w_{1}\overline{x^{2}}&=x^{T}t-w_{0}\overline{x}\\
\end{align*}
$$

Replacing $w_{0}$ in $w_{1}$, we have:

$$
\begin{align*}
w_{1}\overline{x^{2}}&=x^{T}t-\left[\frac{1}{m}(\overline{t}-w_{1}\overline{x})\right]\overline{x}\\
&=x^{T}t-\frac{\overline{x}\overline{t}}{m}+w_{1}\frac{\overline{x}^{2}}{m}\\
\leftrightarrow mw_{1}\overline{x^{2}}&=mx^{T}t-\overline{x}\overline{t}+w_{1}\overline{x}^{2}\\
\leftrightarrow mw_{1}\overline{x^{2}}-w_{1}\overline{x}^{2}&=mx^{T}t-\overline{x}\overline{t}\\
\leftrightarrow w_{1}\left(m\overline{x^{2}}-\overline{x}^{2}\right)&=mx^{T}t-\overline{x}\overline{t}\\
\leftrightarrow w_{1}&=\frac{mx^{T}t-\overline{x}\overline{t}}{m\overline{x^{2}}-\overline{x}^{2}}\\
\end{align*}
$$

The expression for $w_{0}$ becomes tricky if we replace $w_{1}$ in $w_{0}$, so we'll leave it like this.
    
The weights that minimize the mean squared error are:

$$
\begin{align*}
w_{1}&=\frac{mx^{T}t-\overline{x}\overline{t}}{m\overline{x^{2}}-\overline{x}^{2}}\\
w_{0}&=\frac{1}{m}\left (\overline{t}-w_{1}\overline{x}\right)\\
\end{align*}
$$
