"""
MLP for handwritten digit classification, implemented with flax and optax.

Plan:

* any questions from homework?
* installations:
  * `pip install equinox`
* deep learning in jax:
  * dm-haiku, flax.linen, equinox, flax.nnx, try not to rant too much
  * optax and the jax 'ecosystem'
* new jax concept: pytrees (we saw these last time)
* workshop 3 demo:
  * implement MLP with equinox
  * load MNIST https://yann.lecun.com/exdb/mnist/
  * train MLP on MNIST with optax
* challenges:
  * implement a stateful optimiser object from scratch
  * replicate some architectures and performance numbers from lecun's table
"""

import jax
import jax.numpy as jnp

import optax
import equinox

import einops
from jaxtyping import Array, Float, Int, PRNGKeyArray as Key
import tqdm


# # # 
# Architecture


class LinearLayer(equinox.Module):
    weight_matrix: Array
    bias_vector: Array

    def __init__(
        self,
        key: Key,
        num_inputs: int,
        num_outputs: int,
    ):
        # Xavier-initialised weight matrix
        init_bound = jnp.sqrt(6/(num_inputs + num_outputs))
        self.weight_matrix = jax.random.uniform(
            key=key,
            shape=(num_inputs, num_outputs),
            minval=-init_bound,
            maxval=init_bound,
        )

        # zero-initialised bias vector
        self.bias_vector = jnp.zeros((num_outputs,))

    def __call__(
        self,
        x: Float[Array, '... num_inputs'],
    ) -> Float[Array, '... num_outputs']:
        return x @ self.weight_matrix + self.bias_vector


class MLPImageClassifier(equinox.Module):
    layer1: LinearLayer
    layer2: LinearLayer

    def __init__(
        self,
        key: Key,
        image_shape: tuple[int, int],
        num_hidden: int,
        num_classes: int,
    ):
        key_layer1, key_layer2 = jax.random.split(key)
        num_inputs = image_shape[0] * image_shape[1]
        self.layer1 = LinearLayer(key_layer1, num_inputs, num_hidden)
        self.layer2 = LinearLayer(key_layer2, num_hidden, num_classes)

    def __call__(
        self,
        x: Float[Array, '... image_height image_width'],
    ) -> Float[Array, '... num_outputs']:
        x = einops.rearrange(x, '... h w -> ... (h w)')
        x = self.layer1(x)
        x = jnp.tanh(x)
        x = self.layer2(x)
        x = jax.nn.softmax(x, axis=-1)
        return x


# # # 
# Training loop


def main(
    num_hidden: int = 300,
    learning_rate: float = 0.1,
    batch_size: int = 64,
    num_steps: int = 500,
    steps_per_visualisation: int = 16,
    seed: int = 42,
):
    key = jax.random.key(seed)

    # initialise model
    key_model, key = jax.random.split(key)
    model = MLPImageClassifier(
        key=key_model,
        image_shape=(28, 28),
        num_hidden=num_hidden,
        num_classes=10,
    )
    print(model)
    print(model(jnp.zeros((2,28,28))))

    # load and preprocess data
    with jnp.load('mnist.npz') as datafile:
        x_train = jnp.array(datafile['x_train'])
        x_test = jnp.array(datafile['x_test'])
        y_train = jnp.array(datafile['y_train'])
        y_test = jnp.array(datafile['y_test'])
    x_train, x_test = jax.tree.map(lambda x: x/255, (x_train, x_test))
    # TODO: visualise some data!

    # initialise optimiser
    optimiser = optax.sgd(learning_rate)
    optimiser_state = optimiser.init(model)
    # TODO: Try adam, try learning rate decay!

    for step in tqdm.trange(num_steps):
        # sample a batch
        key_batch, key = jax.random.split(key)
        batch = jax.random.choice(
            key=key_batch,
            a=60000,
            shape=(batch_size,),
            replace=False,
        )
        x_batch = x_train[batch]
        y_batch = y_train[batch]

        # compute the batch loss and grad
        loss, grads = jax.value_and_grad(cross_entropy)(
            model,
            x_batch,
            y_batch,
        )

        # compute update, update optimiser and model
        updates, optimiser_state = optimiser.update(grads, optimiser_state)
        model = optax.apply_updates(model, updates)

        # TODO: visualisation! number grid and loss/acc curves!
        if step % steps_per_visualisation == 0:
            acc = accuracy(model, x_test[:1000], y_test[:1000])
            tqdm.tqdm.write(f'train loss: {loss:.3f} | test acc: {acc:.2%}')


def cross_entropy(
    model: MLPImageClassifier,
    x_batch: Float[Array, "b h w"],
    y_batch: Int[Array, "b"],
) -> float:
    """
    Hx(q, p) = - Sum_i p(i) log q(i)
    """
    batch_size, = y_batch.shape
    pred_prob_all_classes = model(x_batch)  # -> batch_size 10
    pred_prob_true_class = pred_prob_all_classes[
        jnp.arange(batch_size),             # for each example
        y_batch,                            # select the prob of the true class
    ]                                       # -> batch_size
    return -jnp.mean(jnp.log(pred_prob_true_class))


def accuracy(
    model: MLPImageClassifier,
    x_batch: Float[Array, "b h w"],
    y_batch: Int[Array, "b"],
) -> float:
    pred_prob_all_classes = model(x_batch)
    highest_prob_class = pred_prob_all_classes.argmax(axis=-1)
    return jnp.mean(y_batch == highest_prob_class)



# # # 
# Visualisation


# TODO: image plotting, my style
# TODO: also a loss plot?


# # # 
# Entry point

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
