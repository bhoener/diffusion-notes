
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
\frac{1}{2} \int_{-\infty}^{\infty}p(x)s_\theta(x)^2dx - \left(
- \int_{-\infty}^{\infty} p(x) \nabla_x s_\theta(x) dx \right) 
$$

