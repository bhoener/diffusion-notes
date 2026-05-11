
### Gaussian diffusion models

There is a forward noising process $q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) \mathbf{I})$ 

This is equivalent to

$$
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon_t; \hspace{1cm} \epsilon_t \sim \mathcal{N}(0, \mathbf{I})
$$


We train diffusion models to learn the reverse process with the following objective:

$$
\mathcal{L}(\theta) =  -p(x_0|x_1) + \sum_t\mathcal{D}_{KL}\left(q^*(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1)}|x_t)\right)
$$
I'm going to assume this is similar to the conditional density in the [[Outlier Video]]:

$$
q^*(x_{t-1}|x_t, x_0) \approx \frac{1}{(2\pi)^{d/2}\sigma^2}e^{-\frac{1}{2\sigma^2}||x_{t-1} - x_t||_2^2}
$$