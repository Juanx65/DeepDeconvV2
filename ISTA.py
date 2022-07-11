import numpy as np
from scipy.linalg import convolution_matrix

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax

from time import time
from functools import partial


class FISTA:
    
    
    def __init__(self, kernel, lam):
        
        A = convolution_matrix(kernel, n=len(kernel), mode="same")
        w, _ = np.linalg.eigh(A.dot(A))

        self.kernel = jnp.expand_dims(-kernel, axis=(0, 1, 2))
        
        self.rho = w.max() / 10.
        self.lam = lam / self.rho
        
        print(self.lam, self.rho)
        
        pass
    
    
    # This decorator is necessary to make JAX play nice with class methods
    # This trick does not work well with @staticmethod decorators
    @partial(jax.jit, static_argnums=(0,))
    def soft(self, x, threshold):
        return jnp.sign(x) * jnp.maximum(jnp.abs(x) - threshold, 0.)
    
    
    @partial(jax.jit, static_argnums=(0,))
    def forward(self, x, kernel):
        y_hat = lax.conv(x, kernel, window_strides=(1, 1), padding="SAME")
        return y_hat
    
    
    @partial(jax.jit, static_argnums=(0,))
    def compute_loss(self, y, x, kernel):
        y_hat = self.forward(x, kernel)
        loss = 0.5 * jnp.square(y - y_hat).mean()
        return loss
    

    @partial(jax.jit, static_argnums=(0,))
    def FISTA_step(self, t, x, r, y):
        
        kernel = self.kernel
        rho = self.rho
        lam = self.lam
        
        loss = self.compute_loss(y, x, kernel)

        grads = jax.grad(self.compute_loss, argnums=1)(y, r, kernel)
        x_new = self.soft(r - rho * grads, lam)
        t_new = 0.5 * (1 + jnp.sqrt(1 + 4 * t**2))
        r_new = x_new + ((t - 1) / t_new) * (x_new - x)

        return loss, t_new, x_new, r_new
    
    
    def solve(self, y, N):
        
        key = jax.random.PRNGKey(int(time()))
        x = jax.random.normal(key, shape=(1, 1,) + y.shape) / y.shape[1]
        r = x.copy()
        t = 1.0
        y_jax = jnp.expand_dims(y, axis=(0, 1))
        
        for i in range(N):
            loss, t, x, r = self.FISTA_step(t, x, r, y_jax)
        
        y_hat = np.array(np.squeeze(self.forward(x, self.kernel)))
        x = np.array(np.squeeze(x))
        
        return loss, x, y_hat

