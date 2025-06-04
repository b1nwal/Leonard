# Leonard
> Named for Leonhard Euler, or Leonard Hofstader. You choose.
## About
Leonard is a neural network built to learn how to take the coordinate of a point in space and construct a set of rotation angles that map to a kinematic chain to produce an appropriate end effector position.

## Inverse Functional Training
$f^{−1}(x)$  
The general idea is that  
> $f^{−1}(f(x))=x$, so  
> loss = lossfn($f^{-1}(f(x))$, $x$)  
> Where $f(x)$  
> is the function that the neural network is trying to learn.

I am calling this Inverse Functional Training, or IFT. I am sure it has a name already.
