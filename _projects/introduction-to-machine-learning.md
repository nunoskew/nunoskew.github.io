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
y(x,\vec{w})=w_{0}+w_{1}x
\end{align*}
$$

## The Error Function

$$
\begin{align*}
E[y(x_{i},\vec{w})]=\frac{1}{2}\sum\limits_{i=1}^{m}(y(x_{i},\vec{w})-t_{i})^{2}
\end{align*}
$$

## Minimizing the Error Function

We want to find the $\underset{\vec{w}}{\operatorname{argmin}} E[y(x_{i},\vec{w})]$ 

$$
\begin{align*}
\underset{\vec{w}}{\operatorname{argmin}} E[y(x_{i},\vec{w})]&\leftrightarrow \left[\frac{dE}{dw}=0\right]_{w}\\
\end{align*}
$$

Let $s_{i}=(y(x_{i},\vec{w})-t_{i})^{2}$ e $r_{i}=y(x_{i},\vec{w})-t_{i}$. 
Lets start by setting up the notation:

$$
\begin{align*}
\frac{dE}{dw}&=\frac{1}{2}*\sum\limits_{i=1}^{m}\frac{ds_{i}}{dw}\\
\frac{ds_{i}}{dw}&=2*r_{i}*\frac{dr_{i}}{dw}\\
\frac{dr_{i}}{dw_{0}}&=1\\
\frac{dr_{i}}{dw_{1}}&=x_{i}\\
\frac{ds_{i}}{dw_{0}}&=2*r_{i}\\
\frac{ds_{i}}{dw_{1}}&=2*r_{i}*x_{i}\\
\end{align*}
$$

Ok, all set!

$$
\begin{align*}
\frac{dE}{dw_{0}}&=\frac{1}{2}\sum\limits_{i=1}^{m}2 r_{i}\\
&=\sum\limits_{i=1}^{m}r_{i}\\
&=\sum\limits_{i=1}^{m} w_{0}+w_{1}x_{i}-t_{i}\\
&=m w_{0}+\sum\limits_{i=1}^{m} w_{1}x_{i}-t_{i}\\
&\rightarrow \frac{dE}{dw}=0\leftrightarrow m w_{0}+\sum\limits_{i=1}^{m} w_{1}x_{i}-t_{i}=0\\
&\rightarrow m w_{0}=\sum\limits_{i=1}^{m} t_{i}-w_{1}x_{i}\\
&\rightarrow m w_{0}=\sum\limits_{i=1}^{m} t_{i}-w_{1}x_{i}\\
&\rightarrow m w_{0}=\sum\limits_{i=1}^{m} t_{i}-\sum\limits_{i=1}^{m} w_{1}x_{i}\\
&\rightarrow m w_{0}=\sum\limits_{i=1}^{m} t_{i}-w_{1}\sum\limits_{i=1}^{m} x_{i}
&\rightarrow w_{0}=\frac{1}{m}\left (\sum\limits_{i=1}^{m} t_{i}-w_{1}\sum\limits_{i=1}^{m} x_{i}\right)
\end{align*}
$$

We also set:

$$
\begin{align*}
\frac{dE}{dw_{1}}&=\frac{1}{2}\sum\limits_{i=1}^{m}2 r_{i} x_{i}\\
&=\sum\limits_{i=1}^{m} x_{i}w_{0}+w_{1}x_{i}^{2}-x_{i}t_{i}\\
&\rightarrow \frac{dE}{w_{1}}=0\leftrightarrow \sum\limits_{i=1}^{m} x_{i}w_{0}+w_{1}x_{i}^{2}-x_{i}t_{i}=0\\
&\rightarrow \sum\limits_{i=1}^{m} w_{1}x_{i}^{2} + \sum\limits_{i=1}^{m} x_{i} w_{0} - x_{i}t_{i} =0\\
&\rightarrow \sum\limits_{i=1}^{m} w_{1} x_{i}^{2}=\sum\limits_{i=1}^{m} x_{i}t_{i} - x_{i}w_{0}\\
&\rightarrow  w_{1}\sum\limits_{i=1}^{m} x_{i}^{2}=\sum\limits_{i=1}^{m} x_{i}t_{i} - x_{i}w_{0}\\
&\rightarrow  w_{1}\sum\limits_{i=1}^{m} x_{i}^{2}=\sum\limits_{i=1}^{m} x_{i}t_{i} - \sum\limits_{i=1}^{m} x_{i}w_{0}
&\rightarrow  w_{1}\sum\limits_{i=1}^{m} x_{i}^{2}=\sum\limits_{i=1}^{m} x_{i}t_{i} - w_{0} \sum\limits_{i=1}^{m} x_{i}
\end{align*}
$$

We now want to find $w_{0}$ e $w_{1}$ that satisfy the following equations: 

$$
\begin{align*}
w_{0}&=\frac{1}{m}\left (\sum\limits_{i=1}^{m} t_{i}-w_{1}\sum\limits_{i=1}^{m} x_{i}\right)\\
w_{1}\sum\limits_{i=1}^{m} x_{i}^{2}&=\sum\limits_{i=1}^{m} x_{i}t_{i} - w_{0} \sum\limits_{i=1}^{m} x_{i}
\end{align*}
$$

The solution looks simpler if we represent through the following terms:

$$
\begin{align*}
\overline{x}&=\sum\limits_{i=1}^{m} x_{i}\\
\overline{x^{2}}&=\sum\limits_{i=1}^{m} x_{i}^{2}\\
\overline{t}&=\sum\limits_{i=1}^{m} t_{i}\\
\overline{xt}&=\sum\limits_{i=1}^{m} x_{i} t_{i}
\end{align*}
$$

Proceeding:

$$
\begin{align*}
w_{0}=\frac{1}{m}\left (\sum\limits_{i=1}^{m} t_{i}-w_{1}\sum\limits_{i=1}^{m} x_{i}\right)&\leftrightarrow w_{0}=\frac{1}{m}\left(\overline{t}-w_{1}\overline{x}\right)\\
w_{1}\sum\limits_{i=1}^{m} x_{i}^{2}=\sum\limits_{i=1}^{m} x_{i}t_{i} - w_{0} \sum\limits_{i=1}^{m} x_{i}&\leftrightarrow w_{1}\overline{x^{2}} = \overline{xt}-w_{0}\overline{x}
\end{align*}
$$

Replacing $w_{0}$ in $w_{1}$, we have:

$$
\begin{align*}
w_{1}\overline{x^{2}}&=\overline{xt}-\frac{1}{m}\left(\overline{t}-w_{1}\overline{x}\right)\\
&=\overline{xt}-\frac{1}{m}\overline{t}+\frac{1}{m} w_{1}\overline{x}\\
&\leftrightarrow m w_{1} \overline{x^{2}} = m\overline{xt} -\overline{t}+w_{1}\overline{x}\\
&\leftrightarrow m w_{1} \overline{x^{2}}- w_{1}\overline{x} = m\overline{xt} -\overline{t}\\
&\leftrightarrow w_{1}\left(m\overline{x^{2}}- \overline{x}\right) = m\overline{xt} -\overline{t}\\
&\leftrightarrow w_{1} = \frac{m\overline{xt} -\overline{t}}{m\overline{x^{2}}- \overline{x}}\\
\end{align*}
$$

Still have to replace $w_{1}$ in $w_{0}$:

$$
\begin{align*}
w_{0}&=\frac{1}{m}(\overline{t}-w_{1}\overline{x})\\
&=\frac{1}{m}(\overline{t}-\frac{m\overline{xt} -\overline{t}}{m\overline{x^{2}}- \overline{x}}\overline{x})\\
&=\frac{1}{m}(\frac{\overline{t}(m\overline{x^{2}}- \overline{x})}{m\overline{x^{2}}- \overline{x}}-\frac{m\overline{xt} -\overline{t}}{m\overline{x^{2}}- \overline{x}}\overline{x})\\
&=\frac{1}{m}(\frac{\overline{t}m\overline{x^{2}}- \overline{t}\overline{x}}{m\overline{x^{2}}- \overline{x}}-\frac{m\overline{x}\overline{xt} -\overline{t}\overline{x}}{m\overline{x^{2}}- \overline{x}})\\
&=\frac{1}{m}(\frac{\overline{t}m\overline{x^{2}}- \overline{t}\overline{x}-m\overline{x}\overline{xt} +\overline{t}\overline{x}}{m\overline{x^{2}}- \overline{x}})\\
&=\frac{1}{m}(\frac{\overline{t}m\overline{x^{2}}-m\overline{x}\overline{xt} +\overline{t}\overline{x}- \overline{t}\overline{x}}{m\overline{x^{2}}- \overline{x}})\\
&=\frac{1}{m}\left[\frac{m(\overline{t}\overline{x^{2}}-\overline{x}\overline{xt})}{m\overline{x^{2}}- \overline{x}}\right]\\
&=\frac{\overline{t}\overline{x^{2}}-\overline{x}\overline{xt}}{m\overline{x^{2}}- \overline{x}}\\
\end{align*}
$$

Finally, we get the expressions for $w_{0}$ and $w_{1}$:

$$
\begin{align*}
w_{0}&=\frac{\overline{t}\overline{x^{2}}-\overline{x}\overline{xt}}{m\overline{x^{2}}- \overline{x}}\\
w_{1} &= \frac{m\overline{xt} -\overline{t}}{m\overline{x^{2}}- \overline{x}}\\
\end{align*}
$$
