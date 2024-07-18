"""
Teacher-student perceptron learning, vanilla JAX.

Notes:

* did anyone try the challenge from last week?
* new dependency `pip install plotille` for easy-ish plotting
* workshop 2 demo: stochastic gradient descent
* challenge 2: multi-layer perceptron!
"""

import time

import plotille
import tqdm
import tyro

import jax
import jax.numpy as jnp


# # # 
# Training loop


def main(
    num_steps: int = 200,
    learning_rate: float = 0.01,
    seed: int = 0,
):
    key = jax.random.key(seed)

    key_init_student, key = jax.random.split(key)
    w = init_params(key_init_student)

    key_init_teacher, key = jax.random.split(key)
    w_star = init_params(key_init_teacher)

    print(vis(student=w, teacher=w_star, overwrite=False))
    print()

    value_and_grad_of_loss = jax.value_and_grad(loss, argnums=(0,1))
    for t in tqdm.trange(num_steps):
        # sample a data point
        key_data_t, key = jax.random.split(key)
        x = jax.random.normal(key_data_t)

        # compute the gradient of the loss wrt. w
        l, (g, g_star) = value_and_grad_of_loss(w, w_star, x)

        # update w
        w = (
            w[0] - learning_rate * g[0],
            w[1] - learning_rate * g[1],
        )
        w_star = (
            w_star[0] - learning_rate * g_star[0],
            w_star[1] - learning_rate * g_star[1],
        )

        # update our visualisation
        tqdm.tqdm.write(vis(student=w, teacher=w_star, x=x))
        tqdm.tqdm.write(
                f"x: {x:+.3f} | loss: {l:.3f} | "
            +f"a: {w[0]:+.3f} | b: {w[1]:+.3f} | "
            +f"a*: {w_star[0]:+.3f} | b*: {w_star[1]:+.3f}"
        )
        time.sleep(0.02)


def loss(w, w_star, x):
    y_student = forward_pass(w, x)
    y_teacher = forward_pass(w_star, x)
    return jnp.mean(
        (y_student - y_teacher)**2
    )


# # # 
# Perceptron architecture


def init_params(key):
    key_a, key_b = jax.random.split(key)
    a = jax.random.normal(key_a)
    b = jax.random.normal(key_b)
    # a, b = jax.random.normal(key, shape=(2,))
    return (a, b)


def forward_pass(w, x):
    a, b = w
    return a * x + b


# # # 
# Visualisation


def vis(x=None, overwrite=True, **models):
    # configure plot
    fig = plotille.Figure()
    fig.width = 40
    fig.height = 15
    fig.set_x_limits(-4, 4)
    fig.set_y_limits(-3, 3)
    
    # compute data and add to plot
    xs = jnp.linspace(-4, 4)
    for (label, w), color in zip(models.items(), ['cyan', 'magenta']):
        ys = forward_pass(w, xs)
        fig.plot(xs, ys, label=label, lc=color)
    
    # add a marker for the input batch
    if x is not None:
        fig.text([x], [0], ['x'], lc='yellow')
    
    # render to string
    figure_str = str(fig.show(legend=True))
    reset = f"\x1b[{len(figure_str.splitlines())+1}A" if overwrite else ""
    return reset + figure_str


# # # 
# Entry point


if __name__ == "__main__":
    tyro.cli(main)
