---
name: Bilinear Interpolation
tools:
image: ../../assets/bilinear-interpolation/esper_machine_latest.gif
description:
relative_url: bilinear_interpolation
---

![alt text](../../assets/bilinear-interpolation/bladerunner_esper_machine.jpg "Blade Runner 1982")
In this scene in the 1982 Blade Runner, Deckard (Harrison Ford) is trying to find more clues about the whereabouts of the remaining replicants.
He does this by inserting a small polaroid picture inside a device, which allows him to zoom in indefinetly while maintaining an apparent high picture definition.
To do this using such a small picture, we would have to artificially increase the resolution of this picture.

<img src="../../assets/bilinear-interpolation/upsampling.svg" style="width: 100%" alt="Upsampling">

In mathematics/computer science this is called upsampling.
To achieve this, we will explore a more general approach called image interpolation, which is just to estimate the image intensity values at unseen coordinates.

The upsampled image was created using the math and code derived in this post.
If you're interested, <em><ins>expand the sections below</ins></em>!
## Linear Interpolation
<details closed>
<summary markdown="span"><em>Problem Setup</em></summary>

The mathematical problem we are trying to solve is the following:
<img src="../../assets/bilinear-interpolation/linear-interpolation-1d.svg" alt="Linear Interpolation 1D" style="width: 100%"/>

Simplifying further the problem, we assume we just know (and care about) the values of the closest neighbors $f(x_{i})$ and $f(x_{i+1})$ to estimate a new value $f(x_{\*})$.
Intuitively, the function value at $x_{\*}$ should be proportional to how close a given neighbor is, so we should started by doing an weighted average of the neighbors' function values implementing the forementioned weight.
</details>
<details closed>
<summary markdown="span"><em>A weighted average of neighbors</em></summary>

$$
\begin{aligned}
f(x_{*})&=(1-\frac{x_{*}-x_{i}}{x_{i+1}-x_{i}})f(x_{i})+(1-\frac{x_{i+1}-x_{*}}{x_{i+1}-x_{i}})f(x_{i+1})
\end{aligned}
$$

Given that the fractions $\frac{x_{\*}-x_{i}}{x_{i+1}-x_{i}}$ and $\frac{x_{i+1}-x_{\*}}{x_{i+1}-x_{i}}$ are normalized distances, i.e. they sum up to one, we can simplify this expression. Ler $d$ be the normalized distance between $x_{\*}$ and $x_{i}$, $d=\frac{x_{\*}-x_{i}}{x_{i+1}-x_{i}}$.

$$
\begin{aligned}
f(x_{*})&=(1-d)f(x_{i})+(1-(1-d))f(x_{i+1})\\
&=(1-d)f(x_{i})+(d)f(x_{i+1})\\
&=f(x_{i})+d(f(x_{i+1})-f(x_{i}))
\end{aligned}
$$

```python
import numpy as np
def find_neighbors(x,new_arg):
    return np.sort(np.argsort(np.abs(x-new_arg))[:2])
def interpolate_1d(x,fx,new_arg):
    x1,x2 = find_neighbors(x,new_arg)
    distance_in_proportion = (new_arg-x[x1])/(x[x2]-x[x1])
    return fx[x1]+distance_in_proportion*(fx[x2]-fx[x1])
fx = np.array([14,2,3,6,7,2,3,6,8])
x=np.linspace(0,len(fx),len(fx))
plt.scatter(x,fx,c=np.repeat(0,len(fx))!=0)
```
<img 
    src="../../assets/bilinear-interpolation/example.svg" 
    alt="synthetic inital example"
    width="600px" />

```python
new_arg_idx = 1
new_arg = 0.5
new_x = np.insert(x,new_arg_idx,new_arg)
new_fx = np.insert(fx,new_arg_idx,interpolate_1d(x,fx,new_arg))
cat = np.repeat('old',len(new_fx))
cat[new_arg_idx] = 'new'
plt.scatter(new_x,new_fx,c=cat!='old')
```

The interpolated point is the yellow one.

<img 
    src="../../assets/bilinear-interpolation/example_part2.svg" 
    alt="synthetic inital example with interpolation"
    width="600px" />
</details>
<details closed>
<summary markdown="span"><em>Geometric Perspective</em></summary>

An equally simple idea would be to draw a straight line between the neighbors' function values, and our estimate would be the intersection to a vertical line drawn at $x_{\*}$.

We will start by representing one dimensional linear interpolation as the solution of a system of linear equations. 
Let $x_{i}$ and $x_{i+1}$ be the nearest neighbors of the value we want to interpolate, $x_{\*}$.

$$
\begin{aligned}
\begin{cases}
f(x_{i})&=m x_{i}+b\\
f(x_{i+1})&=m x_{i+1}+b
\end{cases}
\end{aligned}
$$

This system is represented by the following matrix-vector equation:

$$
\begin{aligned}
\begin{bmatrix}
1&x_{i}\\
1&x_{i+1}
\end{bmatrix}
\begin{bmatrix}
b\\
m
\end{bmatrix}
=
\begin{bmatrix}
f(x_{i})\\
f(x_{i+1})
\end{bmatrix}
\end{aligned}
$$ 

The weights $b$ and $m$ are represented by the equation: 

$$
\begin{aligned}
\begin{bmatrix}
b\\
m
\end{bmatrix}
&=
\begin{bmatrix}
1&x_{i}\\
1&x_{i+1}
\end{bmatrix}^{-1}
\begin{bmatrix}
f(x_{i})\\
f(x_{i+1})
\end{bmatrix}
\\
&=
\frac{1}{x_{i+1}-x_{i}}
\begin{bmatrix}
x_{i+1}&-x_{i}\\
-1&1
\end{bmatrix}
\begin{bmatrix}
f(x_{i})\\
f(x_{i+1})
\end{bmatrix}
\\
&=
\frac{1}{x_{i+1}-x_{i}}
\begin{bmatrix}
x_{i+1}f(x_{i})-x_{i}f(x_{i+1})\\
f(x_{i+1})-f(x_{i})
\end{bmatrix}
\end{aligned}
$$ 

The line equation is:

$$\begin{aligned}
f(x_{*})&=
\frac{f(x_{i+1})-f(x_{i})}{x_{i+1}-x_{i}}x_{*}+
\frac{x_{i+1}f(x_{i})-x_{i}f(x_{i+1})}{x_{i+1}-x_{i}}\\
&=\frac{1}{x_{i+1}-x_{i}}[(f(x_{i+1})-f(x_{i}))x_{*}+x_{i+1}f(x_{i})-x_{i}f(x_{i+1})]
\end{aligned}
$$

``` python
def interpolate_1d_alternative(x,fx,new_arg):
    x1,x2 = find_neighbors(x,new_arg)
    m = ((fx[x2]-fx[x1])/(x[x2]-x[x1]))
    b = (((x[x2]*fx[x1])-(x[x1]*fx[x2]))/(x[x2]-x[x1]))
    return (m*new_arg)+b
new_arg_idx = 1
new_arg = 0.5
new_x = np.insert(x,new_arg_idx,new_arg)
new_fx = np.insert(fx,new_arg_idx,interpolate_1d_alternative(x,fx,new_arg))
cat = np.repeat('old',len(new_fx))
cat[new_arg_idx] = 'new'
plt.scatter(new_x,new_fx,c=cat!='old')
```
<img 
    src="../../assets/bilinear-interpolation/example_part3.svg" 
    alt="synthetic inital example with alternative interpolation"
    width="600px" />

The interpolated point is in the same place as before. 
This leads to the following question:
> **_Question:_**  Is this approach equivalent to the previous one?

Yes, it is!

$$
\begin{align*}  
f(x_{*})&=f(x_{i})+\frac{x_{*}-x_{i}}{x_{i+1}-x_{i}}(f(x_{i+1})-f(x_{i}))\\
&=f(x_{i})+\frac{1}{x_{i+1}-x_{i}}\left(x_{*}f(x_{i+1})-x_{*}f(x_{i})-x_{i}f(x_{i+1})+x_{i}f(x_{i})\right)\\
&=\frac{x_{i+1}f(x_{i})-x_{i}f(x_{i})}{x_{i+1}-x_{i}}+\frac{1}{x_{i+1}-x_{i}}\left(x_{*}f(x_{i+1})-x_{*}f(x_{i})-x_{i}f(x_{i+1})+x_{i}f(x_{i})\right)\\
&=\frac{1}{x_{i+1}-x_{i}}\left(x_{*}f(x_{i+1})-x_{*}f(x_{i})-x_{i}f(x_{i+1})+x_{i}f(x_{i})+x_{i+1}f(x_{i})-x_{i}f(x_{i})\right)\\
&=\frac{1}{x_{i+1}-x_{i}}\left((f(x_{i+1})-f(x_{i}))x_{*}+x_{i+1}f(x_{i})-x_{i}f(x_{i+1})\right)\\
\end{align*}
$$

Hmm.

<img 
    src="../../assets/bilinear-interpolation/walter-sobchak-did-not-know-that.gif"
    alt="big lebowski gif"
    width="600px" />

Now we need to relate the system with the two nearest neighbors, in one dimension, to the system of bilinear interpolation with four closest neighbors in two dimensions.
</details>

## Bilinear Interpolation 

<details closed>
<summary markdown="span"><em>Problem Setup</em></summary>
In the previous section we derived one-dimensional linear interpolation in two different approaches: an intuitive one, the weighted average of the nearest neighbors and an analytical one, draw a line between the nearest neighbors and find the value of that line at the new argument $x^{\*}$.
In this section we are going to generalize these ideas to two dimensions.

In one dimension, the new argument $x^{\*}$ that we are interpolating is surrounded by 2 neighbors, left and right.
Furthermore, we assume that we know the value of the function of these two neighbors.
In a grid of two dimensions, the argument is no longer surround by two neighbors, but by eight.

It would make sense to assume these eight neighbors, but standard bilinear interpolation only assumes 4.

<img src="../../assets/bilinear-interpolation/bilinear_interpolation.svg" style="width: 50%" alt="Bilinear Interpolation Figure">

Lets start by looking into the definition of a bilinear function.
A function defined in $\mathbb{R}^{2}$, $f(x,y)$, is said to be bilinear if $f_{1}(x)$ and $f_{2}(y)$ are linear functions.
An example of a bilinear functions is $f(x,y)=xy$.

We can describe any bilinear function in the following way:

$$
\begin{aligned}
f(x,y)=a+bx+cy+dxy
\end{aligned}
$$

The function has four parameters $a,b,c$ and $d$.
We can find these parameters by solving,

$$
\begin{aligned}
\begin{bmatrix}
1&x_{1}&y_{1}&x_{1}y_{1}\\
1&x_{1}&y_{2}&x_{1}y_{2}\\
1&x_{2}&y_{1}&x_{2}y_{1}\\
1&x_{2}&y_{2}&x_{2}y_{2}\\
\end{bmatrix}
\begin{bmatrix}
a\\
b\\
c\\
d
\end{bmatrix}
=
\begin{bmatrix}
Q_{11}\\
Q_{12}\\
Q_{21}\\
Q_{22}
\end{bmatrix}
\end{aligned}
$$ 
</details>
<details closed>
<summary markdown="span"><em>Solving for the coefficients</em></summary>

Solving for the coefficients, gives a huge expression,

$$
\begin{aligned}
\begin{array}{l}
a=\frac{Q_{11} x_{2} y_{2}}{\left(x_{1}-x_{2}\right)\left(y_{1}-y_{2}\right)}+\frac{Q_{12} x_{2} y_{1}}{\left(x_{1}-x_{2}\right)\left(y_{2}-y_{1}\right)}+\frac{Q_{21} x_{1} y_{2}}{\left(x_{1}-x_{2}\right)\left(y_{2}-y_{1}\right)}+\frac{Q_{22} x_{1} y_{1}}{\left(x_{1}-x_{2}\right)\left(y_{1}-y_{2}\right)} \\
b=\frac{Q_{11} y_{2}}{\left(x_{1}-x_{2}\right)\left(y_{2}-y_{1}\right)}+\frac{Q_{12} y_{1}}{\left(x_{1}-x_{2}\right)\left(y_{1}-y_{2}\right)}+\frac{Q_{21} y_{2}}{\left(x_{1}-x_{2}\right)\left(y_{1}-y_{2}\right)}+\frac{Q_{22} y_{1}}{\left(x_{1}-x_{2}\right)\left(y_{2}-y_{1}\right)} \\
c=\frac{Q_{11} x_{2}}{\left(x_{1}-x_{2}\right)\left(y_{2}-y_{1}\right)}+\frac{Q_{12} x_{2}}{\left(x_{1}-x_{2}\right)\left(y_{1}-y_{2}\right)}+\frac{Q_{21} x_{1}}{\left(x_{1}-x_{2}\right)\left(y_{1}-y_{2}\right)}+\frac{Q_{22} x_{1}}{\left(x_{1}-x_{2}\right)\left(y_{2}-y_{1}\right)} \\
d=\frac{Q_{11}}{\left(x_{1}-x_{2}\right)\left(y_{1}-y_{2}\right)}+\frac{Q_{12}}{\left(x_{1}-x_{2}\right)\left(y_{2}-y_{1}\right)}+\frac{Q_{21}}{\left(x_{1}-x_{2}\right)\left(y_{2}-y_{1}\right)}+\frac{Q_{22}}{\left(x_{1}-x_{2}\right)\left(y_{1}-y_{2}\right)}
\end{array}
\end{aligned}
$$

Now if we represent the neighboring coordinates relative to the ones of the interpolated value then,

$$
\begin{aligned}
(x_{1},y_{1})&=(x-1,y-1)\\
(x_{1},y_{2})&=(x-1,y+1)\\
(x_{2},y_{1})&=(x+1,y-1)\\
(x_{2},y_{2})&=(x+1,y+1)
\end{aligned}
$$

And we perform the inner product with $(1,x,y,xy)$, we get,

$$
f(x,y)=\frac{1}{4}\left(Q_{11}+Q_{21}+Q_{12}+Q_{22}\right)
$$

Which makes sense since inteporlation is just a inverse-distance weighted average of the neighboring values and in this case, they are all at the same distance.

If you want to check the math [follow this link](../../assets/bilinear-interpolation/bilinear-interpolation.pdf).

We are now in good shape to implement Bilinear Interpolation! Moving on to the next and final section, Implementing Bilinear Interpolation.
</details>

<details closed>
<summary markdown="span"><em>Implementation</em></summary>
In the previous section we saw that in order to estimate a pixel intensity through bilinear interpolation using its four closest neighbors, we just compute the average of their intensities. 
<img src="../../assets/bilinear-interpolation/bilinear-interpolation-edge-cases.svg" style="width: 50%" alt="Bilinear Interpolation Edge Cases">
We can safely interpolate 5 but what about 6, 7, 8 and 9?
There are many ways to infer the intensity of the edge cases, but the one it seems more natural to me is to backoff to linear interpolation of 2 neighbors.
This means that to interpolate 6 we are going to average the intensities of 1 and 2, and to interpolate 8, 1 and 3.
This technique is analogous to NLP. 
When we don't have the probability of a certain Ngram, one of the ways to infer it is to compute a convex combination of its context.

To implement this procedure we do,

```python
def interpolate_pixel(mat):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i%2!=0 and j%2!=0:
                mat[i,j]=np.mean([mat[i-1,j-1],mat[i-1,j+1],mat[i+1,j-1],mat[i+1,j+1]])
            elif i%2!=0 and j%2==0:
                mat[i,j]=np.mean([mat[i-1,j],mat[i+1,j]])
            elif i%2==0 and j%2!=0:
                mat[i,j]=np.mean([mat[i,j-1],mat[i,j+1]])
    return mat
```
[Here's a link for the implementation in a jupyter notebook](https://github.com/nunoskew/bilinear-interpolation/blob/master/bilinear-interpolation.ipynb).
</details>
## Links 
* [Wikipedia](https://en.wikipedia.org/wiki/Bilinear_interpolation)
* [Wolfram Mathematica](https://www.wolfram.com/mathematica/)
* [Mathpix](https://mathpix.com/)


