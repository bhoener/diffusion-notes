
https://jxmo.io/posts/variational-autoencoders


## Background

We have a set of examples sampled from a probability distribution $x \sim p(x)$. Behind the scenes, $p(x)$ depends on some latent variable $z$ (apparently, this is just the latent of the model). In order to approximate this distribution, we can model both $p_\theta(z)$ and $p_\theta(x|z)$.

