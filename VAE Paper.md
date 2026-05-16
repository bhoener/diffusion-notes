
https://arxiv.org/pdf/1906.02691



Bayes rule analogy:

>we have a generative model for an earthquake of type A and another for type B, then seeing which of the two describes the data best we can compute a probability for whether earthquake A or B happened.

$$p(A|B) = \frac{p(B|A) \cdot p(A)}{p(B)}$$

General form:

- generative - $p(x|z_L)p(z_L|z_{L-1})\dots p(z_1|z_0)$
- recognition - $q(z_0|z_1)\dots q(z_{L-1}|z_L) q(z_L|X)$

### Conditional models

sometimes, we don't want to find $p(x)$ directly, but instead want $p(y|x)$, which is the probability distribution for $y$ given an observed value of $x$. 

$x$ is called the *input* to the model. (bro why did nobody say this before)

our model is therefore ideally

$$
p_\theta(y|x) \approx p(y|x)
$$

In image classification, $p_\theta(y|x)$ is usually a [categorical distribution](https://en.wikipedia.org/wiki/Categorical_distribution), and i think LLMs use a [multinomial distribution](https://en.wikipedia.org/wiki/Multinomial_distribution). 


### Datasets

If we have a dataset

$$
\mathcal{D} = \{x^{(1)}, x^{(2)}, ... , x^{(n)}\}
$$
We can find the probability assigned to the whole dataset by the model

$$
p_\theta(\mathcal{D}) = \prod_{i=1}^n p_\theta(x^{(i)})
$$
This is just going to be $\approx 0$, so we can look at the log probability instead
$$
\log p_\theta(\mathcal{D}) = \sum_{i=1}^n \log p_\theta(x^{(i)})
$$
since $\log(a \cdot b) = \log(a) + \log(b)$.


### Maximum log-likelihood, SGD

Usually, we use the criterion of maximum log-likelihood for our models. This is equivalent to minimizing kl-divergence between data/model distributions.

We can maximize the log-likelihood using gradient ascent

$$
\nabla_\theta \log p_\theta(D)
$$
but calculating over the whole dataset is usually to expensive. We instead use Stochastic Gradient Descent and sample a random minibatch $\mathcal{M} \subset \mathcal{D}$ and update our parameters with gradients from the following criterion:

$$
\frac{1}{N}\sum_{x \in \mathcal{M}} \log p_\theta(x)
$$
### Intractability

Usually, we cannot find $p_\theta(x) = \int p_\theta(x, z)dz$ because the integral is hard to estimate or calculate directly. However, if we knew $p_\theta(z|x)$, we could use the identity

$$
p_\theta(z|x) = \frac{p_\theta(x, z)}{p_\theta(x)} \implies p_\theta(x) = \frac{p_\theta(x,z)}{p_\theta(z|x)}
$$
Sadly, we don't know $p_\theta(x)$ or $p_\theta(z|x)$ in DLVMs.


## VAEs

### Encoder (approximate posterior)

Since we cannot find $p_\theta(z|x)$ analytically, we attempt to estimate it using an encoder/inference model and train such that

$$
q_\phi(z|x) \approx p_\theta(z|x)
$$
$$
q_\phi(z|x) = q_\phi(z_1...z_N|x) = \prod_{i=1}^N q_\phi(z_i | Pa(z_i), x)
$$
HMMMMMMMMM it seems like we just predict the distribution, not the actual values

$$
(\mu, \log \sigma) = \text{EncoderNeuralNet}_\phi(x)
$$
$$
q_\phi(z|x) = \mathcal{N}(z; \mu, \text{diag}(\sigma))
$$
### ELBO

ELBO is the Evidence (variational) Lower Bound and is the optimization objective for VAEs.

![[Pasted image 20260515090559.png]]

🔥🔥🔥

So we have:

- $p_\theta(z)$ is our prior (gaussian noise)
- $p_\theta(x, z) = p_\theta(z)p_\theta(x|z)$ where $p_\theta(x|z)$ is our decoder
- We have an encoder $q_\phi(z|x)$ which maps dataset samples to the latent space
- We can do some magic inside the latent space and then decode back

For any choice of $q_\phi(z|x)$, we have

$$
\log p_\theta(x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x)]
$$
$$
 = \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x, z)}{p_\theta(z|x)}\right]
$$
$$
= \mathbb{E}_{q_\phi(z|x)}\left[\log \left[\frac{p_\theta(x, z)}{p_\theta(z|x)}\right]\right]
$$
$$
= \mathbb{E}_{q_\phi(z|x)}\left[\log \left[\frac{p_\theta(x, z)}{q_\phi(z|x)} \frac{q_\phi(z|x)}{p_\theta(z|x)}\right]\right]
$$
$$
 = \underbrace{\mathbb{E}_{q_\phi(z|x)} \left[\log \left[\frac{p_\theta(x, z)}{q_\phi(z|x)}\right]\right]}_{ = \mathcal{L}_{\theta, \phi}(x)\text{ (ELBO)}} + \underbrace{\mathbb{E}_{q_\phi(z|x)} \left[\log \left[\frac{q_\phi(z|x)}{p_\theta(z|x)}\right]\right]}_{\mathcal{D}_{KL}(q_\phi(z|x)||p_\theta(z|x))}
$$
So,
$$
\log p_\theta(x) = \mathbb{E}_{q_\phi(z|x)} \left[\log \left[\frac{p_\theta(x, z)}{q_\phi(z|x)}\right]\right] + \mathbb{E}_{q_\phi(z|x)} \left[\log \left[\frac{q_\phi(z|x)}{p_\theta(z|x)}\right]\right]
$$
$$
\implies \mathcal{L}_{\theta, \phi}(x) = \log p_\theta(x) - \mathcal{D}_{KL}(q_\phi(z|x)||p_\theta(z|x)) \le \log p_\theta(x)
$$

The KL divergence describes two things:

1) The divergence between $q_\phi(z|x)$ and $p_\theta(z|x)$
2) The difference between ELBO and $\log p_\theta(x)$

This means that a better approximation of the posterior will lower the gap. Maximizing ELBO will likely push the kl-divergence to zero and $\log p_\theta(x$) upward.

If we want to maximize our objective function, we can use the gradient wrt. the model parameters.

However, it is not generally possible to find $\nabla_{\theta, \phi} \mathcal{L}_{\theta, \phi}(x)$ because of the expectation wrt $q_\phi(z|x)$ in ELBO.

We can approximate it as

$$
\nabla_\theta \mathcal{L}_{\theta, \phi}(x) = \nabla_\theta \mathbb{E}_{q_\phi(z|x)} \left[\log p_\theta(x, z) - \log q_\phi(z|x)\right]
=  \mathbb{E}_{q_\phi(z|x)} \left[\nabla_\theta(\log p_\theta(x, z) - \log q_\phi(z|x))\right]
$$
$$
\approx \nabla_\theta (\log p_\theta (x, z) - \log q_\phi (z|x)) = \nabla_\theta \log p_\theta (x, z)
$$

And for $q_\phi(z|x)$,

$$
\nabla_\phi \mathcal{L}_{\theta, \phi}(x) = \nabla_\phi \mathbb{E}_{q_\phi(z|x)} \left[\log p_\theta(x, z) - \log q_\phi(z|x)\right]
\ne  \mathbb{E}_{q_\phi(z|x)} \left[\nabla_\phi(\log p_\theta(x, z) - \log q_\phi(z|x))\right]
$$
so we must use some tricks.

### Reparameterization trick

We start by expressing $z \sim q_\phi(z|x)$ using an another random variable $\epsilon$. Recall that $q_\phi(z|x)$ is just a probability distribution, so we are not predicting $z$ directly but instead predicting $\mu$ and $\log \sigma$ and sampling noise with those parameters.

We can write

$$
z = g(\epsilon, \phi, x)
$$
![[Pasted image 20260515132719.png]]


Using the change of variable,

$$
\mathbb{E}_{q_\phi(z|x)}\left[f(z)\right] = \int q_\phi(z|x)f(z)dz
$$
Now, according to gemini, $q_\phi(z|x)dz = p(\epsilon)d\epsilon$ because "probability mass must be conserved." Because $z$ is just some deterministic mapping of $\epsilon$, it sort of makes sense that both sides of the equation would represent the same amount of probability mass. I would read further into this, but I don't want to become a statistician.

So, we have

$$
\mathbb{E}_{q_\phi(z|x)}\left[f(z)\right] = \int f(z) q_\phi(z|x)dz = \int f(g(\epsilon, \phi, x))p(\epsilon)d\epsilon = \mathbb{E}_{p(\epsilon)}[f(z)] 
$$
Now,

$$
\nabla_\phi \mathbb{E}_{q_\phi(z|x)}\left[f(z)\right] = \nabla_\phi \mathbb{E}_{p(\epsilon)}[f(z)] = \mathbb{E}_{p(\epsilon)}[\nabla_\phi f(z)] \approx \nabla_\phi f(z)
$$