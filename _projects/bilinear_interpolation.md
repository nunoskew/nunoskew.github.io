---
name: Bilinear Interpolation
tools:
image: https://typesetinthefuture.files.wordpress.com/2016/06/bladerunner_0_43_00_esper_machine.jpg
#alternative: https://typesetinthefuture.files.wordpress.com/2016/06/bladerunner_0_43_00_esper_machine.jpg
description:
relative_url: bilinear_interpolation
---

![alt text](../../assets/bilinear-interpolation/bladerunner_esper_machine.jpg "Blade Runner 1982")
In this scene in the 1982 Blade Runner, Harrison Ford is trying to find more clues about the whereabouts of the remaining replicants.
He does this by inserting a small polaroid picture inside a device, which allows him to zoom in indefinetly while mainting an apparent high picture definition.
To do this using such a small picture, we would have to artificially increase the resolution of this picture.
In mathematics/computer science this is called upsampling.
To achieve this, we will explore a more general approach called image interpolation, which is just to estimate the image intensity values at unseen coordinates.

Linear Interpolation 1D 
======================

The mathematical problem we are trying to solve is the following:
<img src="../../assets/bilinear-interpolation/linear_interpolation_1D.png" alt="drawing" width="600"/>

Simplifying further the problem, we assume we just know (and care about) the values of the closest neighbors $f(x_{i})$ and $f(x_{i+1})$ to estimate a new value $f(x_{\*})$.
Intuitively, the function value at $x_{\*}$ should be proportional to how close a given neighbor is, so we should started by doing an weighted average of the neighbors' function values implementing the forementioned weight.

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
    src="../example.svg" 
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

Bilinear Interpolation 
=====================

In the previous section we derived one-dimensional linear interpolation in two different approaches: an intuitive one, the weighted average of the nearest neighbors and an analytical one, draw a line between the nearest neighbors and find the value of that line at the new argument $x^{\*}$.
In this section we are going to generalize these ideas to two dimensions.

In one dimension, the new argument $x^{\*}$ that we are interpolating is surrounded by 2 neighbors, left and right.
Furthermore, we assume that we know the value of the function of these two neighbors.
In two dimensions the argument is no longer surround by two neighbors, but by eight.

These are the eight neighbors of a point $(i,j)$:

$$
\begin{aligned}
(i-1,j-1)\\
(i-1,j)\\
(i-1,j+1)\\
(i,j-1)\\
(i,j+1)\\
(i+1,j-1)\\
(i+1,j)\\
(i+1,j+1)
\end{aligned}
$$

It would make sense to assume these eight neighbors, but standard bilinear interpolation only assumes 4.
They should be the neighbors in the corners of the square defined by the neighborhood.

Although by intuition we should generalize linear interpolation from one to two dimensions using all of the imediate neighborhood, geometrically it does not make that much sense.
To define a plane, the analogous object of line in three dimensions, we just need three noncoplanar points.
With eight neighbors, we can choose 52 different planes, therefore 52 different linear interpolations. 
These are ${8 \choose 3}$ minus the combinations of points that represent lines, the four sides of the square.
Since we know that bilinear interpolation uses 4 neighbors, this geometrical generalization will not lead us where we want to go.

Lets start by looking into the definition of a bilinear function.
A function defined in $\mathbb{R}^{2}$, $f(x,y)$, is said to be bilinear if $f_{1}(x)$ and $f_{2}(y)$ are linear functions.
An example of a bilinear functions is $f(x,y)=xy$.

In the previous analogy of the plane built from 3 noncoplanar points, we do get the idea that there is a connection between building this plane and a weighted average of the 3 neighbors, generalizing the intuitive approach of the one-dimensional linear interpolation, but we might be wrong.

I think we can describe any bilinear function in the following way:

$$
\begin{aligned}
f(x,y)=a+bx+cy+dxy
\end{aligned}
$$

The function has four parameters. 
If we get four neighbors, we get four equations of four variables, so the system of equations might be determined, i.e. have exactly one possible solution.
If we use matrices to describe this scenario, we can represent it as $Ax=y$ and solve it with $x=A^{-1}y$, assuming that the matrix $A$ has an inverse.
If it doesn't or if we want to use the entire neighborhood composed by the eight points, we solve matrix-vector equation by $x=(A^{T}A)^{-1}A^{T}y$.
Using matrices, it looks like this:

$$
\begin{aligned}
\begin{bmatrix}
1&x_{1}&y_{1}&x_{1}y_{1}\\
1&x_{2}&y_{2}&x_{2}y_{2}\\
1&x_{3}&y_{3}&x_{3}y_{3}\\
1&x_{4}&y_{4}&x_{4}y_{4}\\
1&x_{5}&y_{5}&x_{5}y_{5}\\
1&x_{6}&y_{6}&x_{6}y_{6}\\
1&x_{7}&y_{7}&x_{7}y_{7}\\
1&x_{8}&y_{8}&x_{8}y_{8}\\
\end{bmatrix}
\begin{bmatrix}
a\\
b\\
c\\
d
\end{bmatrix}
=
\begin{bmatrix}
f(x_{1},y_{1})\\
f(x_{2},y_{2})\\
f(x_{3},y_{3})\\
f(x_{4},y_{4})\\
f(x_{5},y_{5})\\
f(x_{6},y_{6})\\
f(x_{7},y_{7})\\
f(x_{8},y_{8})\\
\end{bmatrix}
\end{aligned}
$$

> **_TODO:_** Implemention and test on synthetic data. 

> **_TODO:_** Implementation and test on image. 
