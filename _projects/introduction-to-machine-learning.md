---
name: Regression
tools:
image: ../../assets/intro_ml/chris_bishop_ml_book_smallest.jpg
description:
relative_url: intro_ml 
---

Simple Linear Regression 
======================
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

We want to find $\underset{\vec{w}}{\operatorname{argmin}} E[\vec{w}]$.

Lets start by setting up the notation:

Let $r_{i}=y_{i}(\vec{w})-t_{i}=w_0+w_{1}x_{i}-t_{i}$. 

So $E[r_{i}] =\frac{r_{i}^{2}}{2}$ and $E[\vec{w}]=\frac{1}{2}\sum\limits_{i=1}^{m}r_{i}^{2}$

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

We can verify these expressions empirically. 
Generating points through the line $6+3*x$ with some zero-centered randomlly distributed noise:
<img
    src="../../assets/intro_ml/example.svg"
    alt="randomly generated points"
    width="600px" />

Using the formulas that we derived:

```python
def compute_weights(x,t):
    x_sum,x_squared_sum,t_sum,xt_sum = extract_relevant_variables(x,t)
    w_1 = (len(x)*xt_sum-x_sum*t_sum)/(len(x)*x_squared_sum-(x_sum*x_sum))
    w_0 = (1/len(x))*(t_sum-w_1*x_sum)
    return w_0,w_1
```

<img
    src="../../assets/intro_ml/example_2.svg"
    alt="fitted simple linear regression model"
    width="600px" />
You can try out more functions with different amounts of noise using [this jupyter notebook](https://github.com/nunoskew/simple-linear-regression).

We are now in good shape to go beyond linear functions and proceed into polynomial functions.

Simple Polynomial Regression 
======================

We want to be able to fit any function of $\mathbb{R}$ into $\mathbb{R}$.

Turns out that polynomials are universal approximators. 
Would like to know how to prove such a statement but we'll leave that for later.

## The model
$$
\begin{align*}
y(x,\vec{w})&=\sum\limits_{j=0}^{n}w_{j}x^{j}\\
&=\vec{w}^{T}\vec{x}
\end{align*}
$$

Where $\vec{x}$ is a vector composed by the $n+1$ powers of $x$.

## Minimizing the error function

The error function is still the mean squared error.

To minimize the error function we will have to reset $r_{i}$:

Let $r_{i}=y_{i}(\vec{w})-t_{i}=\left(\sum\limits_{j=0}^{n}w_{j}x_{i}^{j}\right)-t_{i}$. 

So the definitions of $E[r_{i}]$ and $E[\vec{w}]$ remain unchanged.

$$
\begin{align*}
\frac{dE}{dw_{0}}&=\sum\limits_{i=1}^{m} \frac{dE}{dr_{i}}\frac{dr_{i}}{dw_{0}}=\sum\limits_{i=1}^{m}\frac{dE}{dr_{i}}=\sum\limits_{i=1}^{m}r_{i}=\sum\limits_{i=1}^{m}\left(\sum\limits_{j=0}^{n}w_{j}x_{i}^{j}\right)-t_{i}\\
&=\sum\limits_{i=1}^{m} w_{0}+\sum\limits_{i=1}^{m}\sum\limits_{j=1}^{n}w_{j}x_{i}^{j}-t_{i}\\
&=mw_{0}+\sum\limits_{i=1}^{m}\sum\limits_{j=1}^{n}w_{j}x_{i}^{j}-t_{i}=0\leftrightarrow\\
&\leftrightarrow mw_{0}=\sum\limits_{i=1}^{m}t_{i}-\sum\limits_{j=1}^{n}w_{j}x_{i}^{j}\\
&\leftrightarrow w_{0}=\frac{1}{m}\left(\sum\limits_{i=1}^{m}t_{i}-\sum\limits_{j=1}^{n}w_{j}x_{i}^{j}\right)\\
&\leftrightarrow w_{0}=\frac{1}{m}\left(\sum\limits_{i=1}^{m}t_{i}-\vec{w}_{1:j}^{T}\vec{x}_{i}\right)\\
\end{align*}
$$

Now we are going to derive expression for all $w_{i}$ except $w_{0}$.

To do this we need expressions for $\frac{dr_{i}}{dw_{j}}$:

$$
\begin{align*}
\frac{dr_{i}}{dw_{j}}&=x_{i}^{j}\\
\end{align*}
$$

Now we can proceed.

$$
\begin{align*}
\frac{dE}{dw_{j}}&=\frac{1}{2}\sum\limits_{i=1}^{m}2 r_{i} x_{i}^{j}\\
&=\sum\limits_{i=1}^{m}r_{i} x_{i}^{j}\\
&=\sum\limits_{i=1}^{m}\vec{w}^{T}\vec{x}_{i} x_{i}^{j}-t_{i}x_{i}^{j}\\
&=\sum\limits_{i=1}^{m}w_{0}x_{i}^{j}+w_{1}x_{i}^{j+1}+\ldots+w_{n}x_{i}^{n+j}-t_{i}x_{i}^{j}\\
\end{align*}
$$

I can use this equation to verify the expression of $w_{0}$:

$$
\begin{align*}
\frac{dE}{dw_{0}}&=\sum\limits_{i=1}^{m}w_{0}x_{i}^{0}+w_{1}x_{i}^{1}+\ldots+w_{n}x_{i}^{n}-t_{i}x_{i}^{0}\\
&=\sum\limits_{i=1}^{m}w_{0}+w_{1}x_{i}^{1}+\ldots+w_{n}x_{i}^{n}-t_{i}=0\leftrightarrow\\
\leftrightarrow \sum\limits_{i=1}^{m}w_{0}&=\sum\limits_{i=1}^{m} t_{i} - w_{1}x_{i}^{1}-\ldots-w_{n}x_{i}^{n}\leftrightarrow\\
\leftrightarrow mw_{0}&=\sum\limits_{i=1}^{m} t_{i} - w_{1}x_{i}^{1}-\ldots-w_{n}x_{i}^{n}\leftrightarrow\\
\leftrightarrow w_{0}&=\frac{1}{m}\left(\sum\limits_{i=1}^{m}t_{i}-\vec{w}_{1:j}^{T}\vec{x}_{i}\right)\\
\end{align*}
$$

Looks OK! Now i need the expression for an arbitrary parameter $w_{j}$, with $j\neq 0$:

$$
\begin{align*}
\frac{dE}{dw_{j}}&=\sum\limits_{i=1}^{m}w_{0}x_{i}^{j}+\ldots+w_{n}x_{i}^{n+j}-t_{i}x_{i}^{j}\\
&=\sum\limits_{i=1}^{m} w_{j}x_{i}^{2j}+\sum\limits_{i=1}^{m}w_{0}x_{i}^{j}+\ldots+w_{n}x_{i}^{n+j}-t_{i}x_{i}^{j}\\
&=w_{j}\sum\limits_{i=1}^{m} x_{i}^{2j}+\sum\limits_{i=1}^{m}w_{0}x_{i}^{j}+\ldots+w_{n}x_{i}^{n+j}-t_{i}x_{i}^{j}=0\leftrightarrow\\
\leftrightarrow w_{j}\sum\limits_{i=1}^{m} x_{i}^{2j}&=\sum\limits_{i=1}^{m} t_{i}x_{i}^{j} - w_{0}x_{i}^{j}-\ldots-w_{n}x_{i}^{n+j}\leftrightarrow\\
\leftrightarrow w_{j}&=\frac{\sum\limits_{i=1}^{m} t_{i}x_{i}^{j} - w_{0}x_{i}^{j}-\ldots-w_{n}x_{i}^{n+j}}{\sum\limits_{i=1}^{m} x_{i}^{2j}}\\
\leftrightarrow w_{j}&=\frac{\sum\limits_{i=1}^{m} t_{i} - w_{0}x_{i}^{j}-\ldots-w_{n}x_{i}^{j+n}}{\sum\limits_{i=1}^{m} x_{i}^{2j}}\\
\leftrightarrow w_{j}&=\frac{\sum\limits_{i=1}^{m} t_{i} - \vec{w}_{-j}^{T}(x_{i}^{j}\vec{x}_{i})}{\sum\limits_{i=1}^{m} x_{i}^{2j}}\\
\end{align*}
$$

We now have an expression for all model parameters $w_{j}$. 
Now we need to solve the system of $n+1$ equations with $n+1$ unknowns.
How?
If we replace every parameter except $w_{j}$ we get a final expression for $w_{j}$.

The problem that comes up is that the expression for each of the model parameters $w_{j}$ is too complex, and dependent of the number of instances in our dataset, and the degree of the polynomial.
Since linear algebra is the math discipline that is meant to solve these kinds of problems, we will represent the problem as a matrix vector multiplication $A\vec{w}=\vec{t}_{A}$.

We will build the matrix $A$ and the vector $\vec{t}_{A}$ through the representation of $\frac{dE}{dw_j}$:

$$
\begin{align*}
\frac{dE}{dw_{j}}&=\sum\limits_{i=1}^{m}w_{0}x_{i}^{j}+\ldots+w_{n}x_{i}^{n+j}-t_{i}x_{i}^{j}\\
\frac{dE}{dw_{j}}&=w_{0}\sum\limits_{i=1}^{m}x_{i}^{j}+\ldots+w_{n}\sum\limits_{i=1}^{m}x_{i}^{n+j}-\sum\limits_{i=1}^{m}t_{i}x_{i}^{j}=0\leftrightarrow\\
\leftrightarrow&w_{0}\sum\limits_{i=1}^{m}x_{i}^{j}+\ldots+w_{n}\sum\limits_{i=1}^{m}x_{i}^{n+j}=\sum\limits_{i=1}^{m}t_{i}x_{i}^{j}\\
\end{align*}
$$

Through this last equality, we can build a matrix A such that:

$$
\begin{align*}
A=
\begin{bmatrix}
m&\sum\limits_{i=1}^{m}x_{i}&\sum\limits_{i=1}^{m}x_{i}^{2}&\ldots&\sum\limits_{i=1}^{m}x_{i}^{n}\\
\sum\limits_{i=1}^{m}x_{i}&\sum\limits_{i=1}^{m}x_{i}^{2}&\sum\limits_{i=1}^{m}x_{i}^{2}&\ldots&\sum\limits_{i=1}^{m}x_{i}^{n+1}\\
\vdots&\vdots&\vdots&\ldots&\vdots\\
\sum\limits_{i=1}^{m}x_{i}^{n}&\sum\limits_{i=1}^{m}x_{i}^{n+1}&\sum\limits_{i=1}^{m}x_{i}^{n+2}&\ldots&\sum\limits_{i=1}^{m}x_{i}^{2n}\\
\end{bmatrix}
\end{align*}
$$

And a vector $\vec{t}_{A}$ such that:

$$
\begin{align*}
\vec{t}_{A}=
\begin{bmatrix}
\sum\limits_{i=1}^{m}t_{i}\\
\sum\limits_{i=1}^{m}t_{i}x_{i}\\
\sum\limits_{i=1}^{m}t_{i}x_{i}^{2}\\
\vdots\\
\sum\limits_{i=1}^{m}t_{i}x_{i}^{n}\\
\end{bmatrix}
\end{align*}
$$

This way we can finally represent:

$$
\begin{align*}
\frac{dE}{dw_{j}}&=0\leftrightarrow w_{0}\sum\limits_{i=1}^{m}x_{i}^{j}+\ldots+w_{n}\sum\limits_{i=1}^{m}x_{i}^{n+j}=\sum\limits_{i=1}^{m}t_{i}x_{i}^{j}\\
\end{align*}
$$ 

As:

$$
\begin{align*}
\begin{bmatrix}
m&\sum\limits_{i=1}^{m}x_{i}&\sum\limits_{i=1}^{m}x_{i}^{2}&\ldots&\sum\limits_{i=1}^{m}x_{i}^{n}\\
\sum\limits_{i=1}^{m}x_{i}&\sum\limits_{i=1}^{m}x_{i}^{2}&\sum\limits_{i=1}^{m}x_{i}^{2}&\ldots&\sum\limits_{i=1}^{m}x_{i}^{n+1}\\
\vdots&\vdots&\vdots&\ldots&\vdots\\
\vdots&\vdots&\vdots&\ldots&\vdots\\
\sum\limits_{i=1}^{m}x_{i}^{n}&\sum\limits_{i=1}^{m}x_{i}^{n+1}&\sum\limits_{i=1}^{m}x_{i}^{n+2}&\ldots&\sum\limits_{i=1}^{m}x_{i}^{2n}\\
\end{bmatrix}
\begin{bmatrix}
w_{0}\\
w_{1}\\
w_{2}\\
\vdots\\
w_{n}
\end{bmatrix}
=
\begin{bmatrix}
\sum\limits_{i=1}^{m}t_{i}\\
\sum\limits_{i=1}^{m}t_{i}x_{i}\\
\sum\limits_{i=1}^{m}t_{i}x_{i}^{2}\\
\vdots\\
\sum\limits_{i=1}^{m}t_{i}x_{i}^{n}\\
\end{bmatrix}
\leftrightarrow A\vec{w}=\vec{t}_{A}
\end{align*}
$$

Since $A$ is a square matrix and (hopefully) non-singular, we can compute the parameter vector $\vec{w}$ through $\vec{w}=A^{-1}\vec{t}_{A}$.

```python
def compute_A(x,t,n):
    mtx = np.zeros((n+1,n+1))
    for i in range(n+1):
        for j in range(n+1):
            mtx[i,j] = np.sum(np.power(x,(i+j)))
    return mtx

def compute_t_A(x,t,n):
    mtx = np.zeros((n+1,1))
    for j in range(n+1):
        mtx[j,0] = np.dot(np.power(x,j),t)
    return mtx

def extract_polynomial_model_weights(x,t,n):
    A = compute_A(x=x,t=t,n=n)
    t_A = compute_t_A(x=x,t=t,n=n)
    w = np.dot(np.linalg.inv(A),t_A)
    return w

def predict(x,w):
    mtx = np.zeros((len(x),len(w)))
    for i in range(len(x)):
        for j in range(len(w)):
            mtx[i,j] = np.power(x[i],j)
    res = np.dot(mtx,w)
    return res
```

Using the same points as before, we get:
<img
    src="../../assets/intro_ml/example_3.svg"
    alt="fitted simple polynomial regression model"
    width="600px" />
Now we can also fit more complex function such as the logarithm!
<img
    src="../../assets/intro_ml/example_4.svg"
    alt="simple polynomial regression model fitting log"
    width="600px" />

> **_TODO:_** Add link for regression jupyter notebook
