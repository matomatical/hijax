"""
MLP for handwritten digit classification, implemented with equinox and optax.

Workshop plan:

* any questions from homework?
* installations:
  * new library! `pip install equinox`
  * today we'll also embrace `jaxtyping`, `einops`, and `mattplotlib`
* deep learning in jax:
  * dm-haiku, flax.linen, equinox, flax.nnx, try not to rant too much
  * optax and the 'dm jax ecosystem'
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
import draft_mattplotlib as mp


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
        # flatten image
        x = einops.rearrange(x, '... h w -> ... (h w)')
        # apply mlp
        x = self.layer1(x)
        x = jnp.tanh(x)
        x = self.layer2(x)
        # logits -> probability distribution
        x = jax.nn.softmax(x, axis=-1)
        return x


# # # 
# Training loop


def main(
    num_hidden: int = 300,
    learning_rate: float = 0.05,
    batch_size: int = 64,
    num_steps: int = 500,
    steps_per_visualisation: int = 8,
    num_digits_per_visualisation: int = 30,
    seed: int = 42,
):
    key = jax.random.key(seed)

    # initialise model
    print("initialising model...")
    key_model, key = jax.random.split(key)
    model = MLPImageClassifier(
        key=key_model,
        image_shape=(28, 28),
        num_hidden=num_hidden,
        num_classes=10,
    )

    # print(model)
    # print(model(jnp.zeros((2,28,28))))

    print("loading and preprocessing data...")
    with jnp.load('mnist.npz') as datafile:
        x_train = datafile['x_train']
        x_test = datafile['x_test']
        y_train = datafile['y_train']
        y_test = datafile['y_test']
    x_train, x_test, y_train, y_test = jax.tree.map(
        jnp.array,
        (x_train, x_test, y_train, y_test),
    )
    x_train, x_test = jax.tree.map(
        lambda x: x/255,
        (x_train, x_test),
    )

    # print(vis_digits(
    #     digits=x_train[:num_digits_per_visualisation],
    #     true_labels=y_train[:num_digits_per_visualisation],
    #     # pred_labels=model(x_train[:15]).argmax(axis=-1),
    # ))

    print("initialising optimiser...")
    optimiser = optax.sgd(learning_rate) # TODO: try adam, lr_decay
    optimiser_state = optimiser.init(model)

    print("begin training...")
    for step in tqdm.trange(num_steps, dynamic_ncols=True):
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

        # visualisation! number grid, TODO: loss/acc curves!
        if step % steps_per_visualisation == 0:
            acc = accuracy(model, x_test[:1000], y_test[:1000])
            plot = vis_digits(
                digits=x_train[:num_digits_per_visualisation],
                true_labels=y_train[:num_digits_per_visualisation],
                pred_labels=model(
                    x_train[:num_digits_per_visualisation]
                ).argmax(axis=-1),
            )
            tqdm.tqdm.write(
                (f"\x1b[{plot.height+1}A" if step > 0 else "")
                + f"{plot}\ntrain loss: {loss:.3f} | test acc: {acc:.2%}"
            )


def cross_entropy(
    model: MLPImageClassifier,
    x_batch: Float[Array, "b h w"],
    y_batch: Int[Array, "b"],
) -> float:
    """
    Hx(q, p) = - Sum_i p(i) log q(i)
    """
    batch_size, = y_batch.shape
    pred_prob_all_classes = model(x_batch)          # -> batch_size 10
    pred_prob_true_class = pred_prob_all_classes[   # advanced indexing
        jnp.arange(batch_size),                     # for each example
        y_batch,                                    # select prob of true class
    ]                                               # -> batch_size
    log_prob_true_class = jnp.log(pred_prob_true_class)
    avg_cross_entropy = -jnp.mean(log_prob_true_class)
    return avg_cross_entropy


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


def vis_digits(
    digits: Float[Array, "n h w"],
    true_labels: Int[Array, "n"],
    pred_labels: Int[Array, "n"] | None = None,
    downsample: int = 2
) -> mp.plot:
    # downsample images
    ddigits = digits[:,::downsample,::downsample]
    dwidth = digits.shape[2] // downsample

    # if predictions provided, classify as true or false
    if pred_labels is not None:
        corrects = (true_labels == pred_labels)
        cmaps = [None if correct else mp.reds for correct in corrects]
        labels = [f"{t} ({p})" for t, p in zip(true_labels, pred_labels)]
    else:
        cmaps = [None] * len(true_labels)
        labels = [str(t) for t in true_labels]
    array = mp.wrap(*[
        mp.image(ddigit, colormap=cmap)
        ^
        mp.center(mp.text(label), height=2, width=dwidth)
        for ddigit, label, cmap in zip(ddigits, labels, cmaps)
    ], cols=6)
    return array


# # # 
# Entry point

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
