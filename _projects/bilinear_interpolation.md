---
name: Bilinear Interpolation
tools:
image: https://typesetinthefuture.files.wordpress.com/2016/06/bladerunner_0_43_00_esper_machine.jpg
#alternative: https://typesetinthefuture.files.wordpress.com/2016/06/bladerunner_0_43_00_esper_machine.jpg
description:
relative_url: bilinear_interpolation
---

![alt text](https://typesetinthefuture.files.wordpress.com/2016/06/bladerunner_0_43_00_esper_machine.jpg "Blade Runner 1982")
In this scene in the 1982 Blade Runner, Harrison Ford is trying to find more clues about the whereabouts of the remaining replicants.
He does this by inserting a small polaroid picture inside a device, which allows him to zoom in indefinetly while mainting an apparent high picture definition.
To do this using such a small picture we would have to artificially increase the resolution of this picture.
In mathematics/computer science this is called upsampling.
To achieve this, we will have to go through a more general approach called image interpolation, which is just to estimate the image intensity values at unseen coordinates.

Linear Interpolation 1D 
======================

The mathematical problem we are trying to solve is the following:
<img src="../linear_interpolation_1D.png" alt="drawing" width="600"/>

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
```

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
```

Now we need to relate the system with the two nearest neighbors, in one dimension, to the system of bilinear interpolation with four closest neighbors in two dimensions.

Bilinear Interpolation 
=====================

Work in progress.
