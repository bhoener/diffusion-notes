
![[Pasted image 20260426164014.png]]

We know how to add noise to an image via $q(x_t | x_{t-1})$. We can train our model to predict the less noisy image $p_\theta(x_{t-1}|x_t)$, but don't know the underlying function $q(x_{t-1}|x_t)$. 

### Reparameterization trick

