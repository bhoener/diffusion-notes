
https://www.youtube.com/watch?v=B4oHJpEJBAA&t=803s


We have a dataset sampled from an underlying probability distribution $x \sim p(x)$. We do not know this distribution, but we want to model it with a function $f_\theta(x)$. We let

$$
p_\theta(x) = \frac{\overbrace{e^{-f_\theta(x)}}^{\text{keep positive}}}{\underbrace{Z_\theta}_{\text{normalize}}}
$$

However, we cannot easily find $Z_\theta$ without computing the integral for all samples in the dataset.

### Score Matching

We look at the score: $\nabla_x \log p_\theta(x)$ 

$$
\nabla_x \log p_\theta(x) = \nabla_x \log \frac{e^{-f_\theta(x)}}{Z_\theta} = \nabla_x \log e^{-f_\theta(x)} - \cancelto{0}{\nabla_x \log Z_\theta} = \nabla_x -f_\theta(x) = s_\theta(x)
$$

This tells us the direction of steepest ascent towards data points.

We want to minimize:

$$
\frac{1}{2}\mathbb{E}_{p(x)}[||\nabla_x \log p(x) - s_\theta(x)||_2^2] = \frac{1}{2}\int_{-\infty}^{\infty} p(x) (\nabla_x \log p(x) - s_\theta(x))^2dx
$$

since

$$
 \mathbb{E}_{p(x)}[f(x)] = \int f(x) p(x) dx
$$

For simplicity, we assume $\nabla_x \log p(x) - s_\theta(x)$ is a scalar so that the squared $l_2$ norm just becomes a regular square.

We can further expand this integral:

$$
... = \frac{1}{2}\int_{-\infty}^{\infty}p(x)\left([\nabla_x \log p(x)]^2 - [s_\theta(x)]^2 - 2\nabla_x \log p(x) s_\theta(x)\right)dx
$$
$$
=\frac{1}{2} \int_{-\infty}^{\infty}p(x) \nabla_x \log p(x)^ 2dx +  \frac{1}{2} \int_{-\infty}^{\infty}p(x)s_\theta(x)^2dx - \int_{-\infty}^{\infty}p(x) \nabla_x \log p(x) s_\theta(x)dx
$$
Note that by the chain rule,

$$
\nabla_x \log p(x) = \frac{\nabla_x p(x)}{p(x)}
$$

This gives

$$
\frac{1}{2} \int_{-\infty}^{\infty}p(x) \nabla_x \log p(x)^ 2dx +  \frac{1}{2} \int_{-\infty}^{\infty}p(x)s_\theta(x)^2dx - \int_{-\infty}^{\infty}p(x) \frac{\nabla_x p(x)}{p(x)}  s_\theta(x)dx
$$
$$
\frac{1}{2} \int_{-\infty}^{\infty}p(x) \nabla_x \log p(x)^ 2dx +  \frac{1}{2} \int_{-\infty}^{\infty}p(x)s_\theta(x)^2dx - \int_{-\infty}^{\infty} \nabla_x p(x)  s_\theta(x)dx
$$

Here, we can apply integration by parts to the last term:

$$
\int u \cdot dv = u\cdot v - \int v\cdot du
$$

In this case, we let 
$$
u = s_\theta(x) \hspace{1cm} dv=\nabla_x p(x)
$$
$$
v = p(x) \hspace{1cm} du = \nabla_x s_\theta(x) dx
$$
$$
\implies \frac{1}{2} \int_{-\infty}^{\infty}p(x) \nabla_x \log p(x)^ 2dx +  \frac{1}{2} \int_{-\infty}^{\infty}p(x)s_\theta(x)^2dx - \left(\underbrace{\cancelto{0}{s_\theta(x) p(x) \biggr\rvert_{-\infty}^{\infty}}}_{\text{zero at infinity}} - \int_{-\infty}^{\infty} p(x) \nabla_x s_\theta(x) dx \right) 
$$

Notice that the first term is constant wrt the model parameters, so it can be ignored for optimization and we are left with:

$$
\frac{1}{2} \int_{-\infty}^{\infty}p(x)s_\theta(x)^2dx +
 \int_{-\infty}^{\infty} p(x) \nabla_x s_\theta(x) dx
$$

Using the formula for expectation $\mathbb{E}_{p(x)}= \int_{-\infty}^{\infty}p(x)dx$, we get

$$
\frac{1}{2} \mathbb{E}_{p(x)}[s_\theta(x)^2] + \underbrace{\mathbb{E}_{p(x)}[\nabla_x s_\theta(x)]}_{\text{difficult to compute}}
$$
### Training Issues

When we train, most of the time we will be close to the datapoints so the scores will be mostly accurate. However, there could be large spaces of low probability in between datapoints where scores contain little information. In order to help resolve this, we add some noise to all the datapoints.

$$
\tilde{x} = x + \epsilon; \hspace{1cm} \epsilon \sim \mathcal{N}(0, \sigma^2I)
$$The new distribution is
$$
p(x) \longrightarrow p_\sigma(\tilde{x})
$$
And the corresponding new objective:

$$
\frac{1}{2} \mathbb{E}_{p_\sigma(\tilde{x})}[||\nabla_\tilde{x} \log p_\sigma(\tilde{x}) - s_\theta(\tilde{x})||_2^2]
$$
### New Objective

We can use ideas from denoising autoencoders to get an even better objective function.

We start by rewriting the objective expectation as an integral:

$$
\frac{1}{2} \mathbb{E}_{p_\sigma(\tilde{x})}[||\nabla_\tilde{x} \log p_\sigma(\tilde{x}) - s_\theta(\tilde{x})||_2^2] = \frac{1}{2} \int_{-\infty}^{\infty}p_\sigma(\tilde{x})\left(\nabla_\tilde{x} \log p_\sigma(\tilde{x}) - s_\theta(\tilde{x}) \right)^2 d\tilde{x}
$$
Expanding:

$$
 ... = \frac{1}{2} \int_{-\infty}^{\infty}p_\sigma(\tilde{x}) [\nabla_\tilde{x} \log p_\sigma(\tilde{x})]^2d\tilde{x} + \frac{1}{2} \int_{-\infty}^{\infty}p_\sigma(\tilde{x})[s_\theta(\tilde{x})]^2d\tilde{x} - \int_{-\infty}^{\infty}p_\sigma(\tilde{x})\nabla_\tilde{x} \log p_\sigma(\tilde{x}) s_\theta(\tilde{x})d\tilde{x}
$$
Using $\nabla_\tilde{x} \log p_\sigma(\tilde{x}) = \frac{\nabla_\tilde{x} p_\sigma(\tilde{x})}{p_\sigma(\tilde{x})}$, we get
$$
\frac{1}{2} \int_{-\infty}^{\infty}p_\sigma(\tilde{x}) [\nabla_\tilde{x} \log p_\sigma(\tilde{x})]^2d\tilde{x} + \frac{1}{2} \int_{-\infty}^{\infty}p_\sigma(\tilde{x})[s_\theta(\tilde{x})]^2d\tilde{x} - \int_{-\infty}^{\infty}\cancel{p_\sigma(\tilde{x})}\frac{\nabla_\tilde{x} p_\sigma(\tilde{x})}{\cancel{p_\sigma(\tilde{x})}} s_\theta(\tilde{x})d\tilde{x}
$$

Apparently, we can use a concept called marginalization to turn $p_\sigma(\tilde{x})$ into something nicer:

$$
p_\sigma(\tilde{x}) = \int p(x) \hspace{1mm} p_\sigma(\tilde{x}|x) \hspace{1mm} dx
$$
So, the expression from before becomes:

$$
\frac{1}{2} \int_{-\infty}^{\infty}p_\sigma(\tilde{x}) [\nabla_\tilde{x} \log p_\sigma(\tilde{x})]^2d\tilde{x} + \frac{1}{2} \int_{-\infty}^{\infty}p_\sigma(\tilde{x})[s_\theta(\tilde{x})]^2d\tilde{x} - \int_{-\infty}^{\infty}\nabla_\tilde{x} (\int p(x) p_\sigma(\tilde{x}|x)dx) \cdot s_\theta(\tilde{x})d\tilde{x}
$$

Because $p(x)$ does not depend on $\tilde{x}$, the $\nabla_\tilde{x}$ can be moved inside the integral as follows:

$$
\frac{1}{2} \int_{-\infty}^{\infty}p_\sigma(\tilde{x}) [\nabla_\tilde{x} \log p_\sigma(\tilde{x})]^2d\tilde{x} + \frac{1}{2} \int_{-\infty}^{\infty}p_\sigma(\tilde{x})[s_\theta(\tilde{x})]^2d\tilde{x} - \int_{-\infty}^{\infty} (\int_x p(x) \nabla_\tilde{x} p_\sigma(\tilde{x}|x)dx) \cdot s_\theta(\tilde{x})d\tilde{x}
$$

Using a weird trick,
$$
\nabla_\tilde{x} \log p_\sigma(\tilde{x}|x) = \frac{\nabla_\tilde{x}p_\sigma(\tilde{x}|x)}{p_\sigma(\tilde{x}|x)} \implies \nabla_\tilde{x}p_\sigma(\tilde{x}|x) = p_\sigma(\tilde{x}|x)\nabla_\tilde{x}\log p_\sigma(\tilde{x}|x)
$$
$$
... = \frac{1}{2} \int_{-\infty}^{\infty}p_\sigma(\tilde{x}) [\nabla_\tilde{x} \log p_\sigma(\tilde{x})]^2d\tilde{x} + \frac{1}{2} \int_{-\infty}^{\infty}p_\sigma(\tilde{x})[s_\theta(\tilde{x})]^2d\tilde{x} - \int_{-\infty}^{\infty} (\int_{-\infty}^{\infty} p(x) p_\sigma(\tilde{x}|x)\nabla_\tilde{x}\log p_\sigma(\tilde{x}|x)dx) \cdot s_\theta(\tilde{x})d\tilde{x}
$$$$
= \frac{1}{2} \int_{-\infty}^{\infty}p_\sigma(\tilde{x}) [\nabla_\tilde{x} \log p_\sigma(\tilde{x})]^2d\tilde{x} + \frac{1}{2} \int_{-\infty}^{\infty}p_\sigma(\tilde{x})[s_\theta(\tilde{x})]^2d\tilde{x} - \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} p(x) p_\sigma(\tilde{x}|x)\nabla_\tilde{x}\log p_\sigma(\tilde{x}|x) \cdot s_\theta(\tilde{x}) dx d\tilde{x}
$$
Turning this back into expectations:

$$
= \frac{1}{2} \mathbb{E}_{\tilde{x} \sim p_\sigma(\tilde{x})} [||\nabla_\tilde{x} \log p_\sigma(\tilde{x})||_2^2] + \frac{1}{2} \mathbb{E}_{\tilde{x} \sim p_\sigma(\tilde{x})} [||s_\theta(\tilde{x})||_2^2] - \mathbb{E}_{x \sim p(x), \tilde{x} \sim p_\sigma(\tilde{x}|x)} [\nabla_\tilde{x}\log p_\sigma(\tilde{x}|x) \cdot s_\theta(\tilde{x})]
$$
Using marginalization again,

$$
= \frac{1}{2} \mathbb{E}_{\tilde{x} \sim p_\sigma(\tilde{x})} [||\nabla_\tilde{x} \log p_\sigma(\tilde{x})||_2^2] + \frac{1}{2} \mathbb{E}_{x \sim p(x), \tilde{x} \sim p_\sigma(\tilde{x|x})} [||s_\theta(\tilde{x})||_2^2] - \mathbb{E}_{x \sim p(x), \tilde{x} \sim p_\sigma(\tilde{x}|x)} [\nabla_\tilde{x}\log p_\sigma(\tilde{x}|x) \cdot s_\theta(\tilde{x})]
$$
$$
= \frac{1}{2} \mathbb{E}_{\tilde{x} \sim p_\sigma(\tilde{x})} [||\nabla_\tilde{x} \log p_\sigma(\tilde{x})||_2^2] + \frac{1}{2} \mathbb{E}_{x \sim p(x), \tilde{x} \sim p_\sigma(\tilde{x|x})} [||s_\theta(\tilde{x})||_2^2  - 2 \nabla_\tilde{x}\log p_\sigma(\tilde{x}|x) \cdot s_\theta(\tilde{x})]
$$

Now, because $(a - b)^2 = a^2 + b^2 -2ab$, we can add and subtract a $||\nabla_\tilde{x} \log p_\sigma(\tilde{x}|x)||_2^2$ to "simplify".

$$
\frac{1}{2} \mathbb{E}_{\tilde{x} \sim p_\sigma(\tilde{x})} [||\nabla_\tilde{x} \log p_\sigma(\tilde{x})||_2^2] + \frac{1}{2} \mathbb{E}_{x \sim p(x), \tilde{x} \sim p_\sigma(\tilde{x|x})} [||s_\theta(\tilde{x})||_2^2  - 2 \nabla_\tilde{x}\log p_\sigma(\tilde{x}|x) \cdot s_\theta(\tilde{x}) + ||\nabla_\tilde{x} \log p_\sigma(\tilde{x}|x)||_2^2 - ||\nabla_\tilde{x} \log p_\sigma(\tilde{x}|x)||_2^2]
$$

$$
 = \frac{1}{2} \mathbb{E}_{\tilde{x} \sim p_\sigma(\tilde{x})} [||\nabla_\tilde{x} \log p_\sigma(\tilde{x})||_2^2] + \frac{1}{2} \mathbb{E}_{x \sim p(x), \tilde{x} \sim p_\sigma(\tilde{x|x})} [||s_\theta(\tilde{x}) - \nabla_\tilde{x} \log p_\sigma(\tilde{x}|x)||_2^2 - ||\nabla_\tilde{x} \log p_\sigma(\tilde{x}|x)||_2^2]
$$
We can separate the last term from the expectation:

$$
 \underbrace{\cancel{\frac{1}{2} \mathbb{E}_{\tilde{x} \sim p_\sigma(\tilde{x})} [||\nabla_\tilde{x} \log p_\sigma(\tilde{x})||_2^2]}}_\text{constant} + \frac{1}{2} \mathbb{E}_{x \sim p(x), \tilde{x} \sim p_\sigma(\tilde{x|x})} [||s_\theta(\tilde{x}) - \nabla_\tilde{x} \log p_\sigma(\tilde{x}|x)||_2^2 ] - \underbrace{\cancel{\frac{1}{2}\mathbb{E}_{x \sim p(x) \tilde{x} \sim p(\tilde{x}|x)}[||\nabla_\tilde{x} \log p_\sigma(\tilde{x}|x)||_2^2]}}_\text{constant}
$$
$$
\implies \frac{1}{2} \mathbb{E}_{x \sim p(x), \tilde{x} \sim p_\sigma(\tilde{x|x})} [||s_\theta(\tilde{x}) - \nabla_\tilde{x} \log p_\sigma(\tilde{x}|x)||_2^2 ]
$$
### Conditional density

When we add noise to $x$ to obtain $\tilde{x}$, we can get the conditional density (what I assume to be the probability $p_\sigma(\tilde{x}|x))$ given the specific pair $(x, \tilde{x})$). 

$$
p_\sigma(\tilde{x}|x) = \frac{1}{(2\pi)^{d/2}\sigma^2}e^{-\frac{1}{2\sigma^2}||\tilde{x} - x||_2^2}
$$
Remember that we want to calculate $\nabla_\tilde{x} \log p_\sigma(\tilde{x}|x)$.

$$
\nabla_\tilde{x} \log p_\sigma(\tilde{x}|x) = \nabla_\tilde{x} \log \frac{1}{(2\pi)^{d/2}\sigma^2}e^{-\frac{1}{2\sigma^2}||\tilde{x} - x||_2^2}
$$
$$
= \cancelto{0}{\nabla_\tilde{x} \log \frac{1}{(2\pi)^{d/2}\sigma^2}} + \nabla_\tilde{x} \log e^{-\frac{1}{2\sigma^2}||\tilde{x} - x||_2^2}
$$
$$
 = -\frac{1}{2\sigma^2}\nabla_\tilde{x}||\tilde{x} - x||_2^2
$$
By the chain rule,
$$
 -\frac{1}{2\sigma^2}\nabla_\tilde{x}||\tilde{x} - x||_2^2 = -\frac{1}{2\sigma^2} \cdot 2(\tilde{x} - x) = -\frac{1}{\sigma^2} (\tilde{x} - x) = \frac{1}{\sigma^2}(x - \tilde{x})
$$
But recall that $\tilde{x} = x + \epsilon$, so:
$$
\frac{1}{\sigma^2}(x - \tilde{x}) = \frac{1}{\sigma^2}(x - x - \epsilon) = -\frac{1}{\sigma^2}\epsilon
$$
Our objective is now to minimize:

$$
\frac{1}{2} \mathbb{E}_{x \sim p(x), \tilde{x} \sim p_\sigma(\tilde{x|x})} [||s_\theta(\tilde{x}) -  (-\frac{1}{\sigma^2}\epsilon)||_2^2 ]
$$

### Summary of training

We started by wanting to estimate the underlying data distribution $p(x)$ using our model $p_\theta(x)$. This is difficult to do directly, so we use the score: $\nabla_x \log p(x)$.

Using math tricks, we can reduce this down, but we eventually still get something difficult to use in practice.

We steal ideas from denoising autoencoders and add noise to all the datapoints giving us the new distribution $p_\sigma(\tilde{x})$. This idea eventually reduces down to a simple training objective: having the scores match the negative of the noise. 

### Sampling

We cannot simply use the score once, as it will likely over-or under-shoot the desired outcome. We can instead use a simple method: move a tiny bit in the score's direction each step for $K$ steps. 

$$
\tilde{x}_{i+1} = \tilde{x}_i + \alpha \cdot s_\theta(\tilde{x_i})
$$
Where $\alpha$ is a scaling factor and $i = 0, 1, 2, ..., K$.

However, this converges only to the mean of the dataset. We can get better results by using $\text{Langevin Dynamics Sampling}$.

$$
\tilde{x}_{i+1} = \tilde{x}_i + \alpha \cdot s_\theta(\tilde{x}_i) + \sqrt{2\alpha} \cdot \underbrace{\epsilon}_\text{noise}
$$

### Multiple noise levels

In order to solve the issue of too much noise causing loss of information from the original distribution and too little causing not enough coverage of datapoints, we can simply vary the noise throughout training examples.

We can also give this noise value to our model to help it in training.

$$
s_\theta(\tilde{x}) \rightarrow s_\theta(\tilde{x}, \sigma_t)
$$

### SDEs

I might have a skill issue here


Stochastic differential equations can be used to model processes with randomness that change over time.

One form is:

$$
dx = \underbrace{f(x, t)}_\text{drift coeff. }dt + \underbrace{g(t)}_\text{diffusion coeff.} \overbrace{dw}^\text{change in noise}
$$Our process is

$$
\tilde{x} = x + \epsilon; \hspace{1cm} \epsilon \sim \mathcal{N}(0, \sigma_t^2I)
$$There is no drift, so we just have

$$
\underbrace{dx = g(t)dw}_\text{Forward SDE}
$$
In theory, we can solve this and get a formula for sampling

