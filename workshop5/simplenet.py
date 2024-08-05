"""
CNN for handwritten digit classification, implemented with equinox and optax.
"""

from typing import Literal
from jaxtyping import Array, Float, Int, PRNGKeyArray

import jax
import jax.numpy as jnp
import einops
import optax
import equinox as eqx

import tqdm
import mattplotlib as mp


# # # 
# Architecture


class Subsample2x2(eqx.Module):
    weights: Float[Array, "c"]
    biases: Float[Array, "c"]


    def __init__(self, num_channels: int):
        self.weights = jnp.ones((num_channels, 1, 1)) / 4
        self.biases = jnp.zeros((num_channels, 1, 1))


    def __call__(self, x: Float[Array, "c h w"]) -> Float[Array, "c h w"]:
        sums = einops.reduce(x, 'c (h 2) (w 2) -> c h w', 'sum')
        return self.weights * sums + self.biases


class SimpLeNet(eqx.Module):
    C1: eqx.nn.Conv2d
    S2: Subsample2x2
    C3: eqx.nn.Conv2d
    S4: Subsample2x2
    C5: eqx.nn.Conv2d
    F6: eqx.nn.Linear
    Out: eqx.nn.Linear


    def __init__(self, key: PRNGKeyArray):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.C1 = eqx.nn.Conv2d(1, 6, kernel_size=5, padding=2, key=k1)
        self.S2 = Subsample2x2(num_channels=6)
        self.C3 = eqx.nn.Conv2d(6, 16, kernel_size=5, padding=0, key=k2)
        self.S4 = Subsample2x2(num_channels=16)
        self.C5 = eqx.nn.Conv2d(16, 120, kernel_size=5, padding=0, key=k3)
        self.F6 = eqx.nn.Linear(120, 84, key=k4)
        self.Out = eqx.nn.Linear(84, 10, key=k5)


    def forward(
        self,
        image: Float[Array, "28 28"],
    ) -> Float[Array, "10"]:
        # Input:         1x28x28
        x = einops.rearrange(image, 'h w -> 1 h w')
        # C1:       ->   6x28x28
        x = scaled_tanh(self.C1(x))
        # S2:       ->   6x14x14
        x = scaled_tanh(self.S2(x))
        # C3*:      ->  16x10x10 (note: fully connected channels)
        x = scaled_tanh(self.C3(x))
        # S4:       ->  16x5x5
        x = scaled_tanh(self.S4(x))
        # C5:       -> 120x1x1 (note: equiv. dense 400->120 at this size)
        x = scaled_tanh(self.C5(x))
        # (flatten) -> 120
        x = jnp.ravel(x)
        # F6:       -> 84
        x = scaled_tanh(self.F6(x))
        # Output*:  -> 10 (note: learned map, no hand-made RBF code)
        x = self.Out(x)
        return jax.nn.softmax(x)


    def forward_batch(
        self,
        x_batch: Float[Array, "b 28 28"],
    ) -> Float[Array, "b 10"]:
        return jax.vmap(self.forward)(x_batch)


def scaled_tanh(x):
    return 1.7159 * jnp.tanh(0.6667 * x)


# # # 
# Training loop


def main(
    learning_rate: float = 0.05,
    lr_schedule: bool = False,
    opt: Literal["sgd", "adam", "adamw"] = "sgd",
    batch_size: int = 512,
    num_steps: int = 128,
    steps_per_visualisation: int = 4,
    num_digits_per_visualisation: int = 10,
    seed: int = 42,
):
    key = jax.random.key(seed)


    print("initialising model...")
    key_model, key = jax.random.split(key)
    model = SimpLeNet(key=key_model)

    # print(model)

    
    print("loading and preprocessing data...")
    with jnp.load('mnist.npz') as datafile:
        x_train = jnp.array(datafile['x_train'])
        x_test = jnp.array(datafile['x_test'])
        y_train = jnp.array(datafile['y_train'])
        y_test = jnp.array(datafile['y_test'])
    x_train = 1.275 * x_train / 255 - 0.1
    x_test = 1.275 * x_test / 255 - 0.1

    # print(model.forward(x_train[0]))
    

    print("initialising optimiser...")
    # configure the optimiser
    if lr_schedule:
        learning_rate = optax.linear_schedule(
            init_value=learning_rate,
            end_value=learning_rate/100,
            transition_steps=num_steps,
        )
    if opt == 'sgd':
        optimiser = optax.sgd(learning_rate)
    elif opt == 'adam':
        optimiser = optax.adam(learning_rate)
    elif opt == 'adamw':
        optimiser = optax.adamw(learning_rate)
    # initialise the optimiser state
    opt_state = optimiser.init(model)
    
    # print(opt_state)


    print("begin training...")
    losses = []
    accuracies = []
    for step in tqdm.trange(num_steps, dynamic_ncols=True):
        # sample a batch
        key_batch, key = jax.random.split(key)
        batch = jax.random.choice(
            key=key_batch,
            a=y_train.size,
            shape=(batch_size,),
            replace=False,
        )
        x_batch = x_train[batch]
        y_batch = y_train[batch]

        # compute the batch loss and grad
        loss, grads = jax.value_and_grad(batch_cross_entropy)(
            model,
            x_batch,
            y_batch,
        )

        # compute update, update optimiser and model
        updates, opt_state = optimiser.update(grads, opt_state, model)
        model = optax.apply_updates(model, updates)

        # track metrics
        losses.append((step, loss))
        test_acc = batch_accuracy(model, x_test[:1000], y_test[:1000])
        accuracies.append((step, test_acc))

        # visualisation!
        if step % steps_per_visualisation == 0 or step == num_steps - 1:
            digit_plot = vis_digits(
                digits=x_test[:num_digits_per_visualisation],
                true_labels=y_test[:num_digits_per_visualisation],
                model=model,
            )
            metrics_plot = vis_metrics(
                losses=losses,
                accuracies=accuracies,
                total_num_steps=num_steps,
            )
            plot = digit_plot ^ metrics_plot
            if step == 0:
                tqdm.tqdm.write(str(plot))
            else:
                tqdm.tqdm.write(f"\x1b[{plot.height}A{plot}")


# # # 
# Metrics


def batch_cross_entropy(
    model: SimpLeNet,
    x_batch: Float[Array, "b h w"],
    y_batch: Int[Array, "b"],
) -> float:
    """
    Average cross entropy from across the batch.
    """
    vmapped_cross_entropy_fn = jax.vmap(
        cross_entropy,
        in_axes=(None,0,0),
    )
    all_cross_entropies = vmapped_cross_entropy_fn(
        model,
        x_batch,
        y_batch,
    )
    avg_cross_entropy = all_cross_entropies.mean()
    return avg_cross_entropy


def cross_entropy(
    model: SimpLeNet,
    x: Float[Array, "h w"],
    y: int,
) -> float:
    # Cross entropy formula: Hx(q, p) = - Sum_i p(i) log q(i)
    return -jnp.log(model.forward(x)[y])


def batch_accuracy(
    model: SimpLeNet,
    x_batch: Float[Array, "b h w"],
    y_batch: Int[Array, "b"],
) -> float:
    pred_prob_all_classes = model.forward_batch(x_batch)
    highest_prob_class = pred_prob_all_classes.argmax(axis=-1)
    return jnp.mean(y_batch == highest_prob_class)


# # # 
# Visualisation


def vis_digits(
    digits: Float[Array, "n h w"],
    true_labels: Int[Array, "n"],
    model: SimpLeNet | None = None,
) -> mp.plot:
    # shrink and normalise images
    ddigits = einops.reduce(
        (digits + 0.1) / 1.275,
        'b (h 2) (w 2) -> b h w',
        'mean',
    )
    width = ddigits.shape[-1]

    # if model is provided, classify digits and mark correct or incorrect
    if model is not None:
        pred_labels = model.forward_batch(digits).argmax(axis=-1)
        corrects = (true_labels == pred_labels)
        cmaps = [None if correct else mp.reds for correct in corrects]
        labels = [f"{t} ({p})" for t, p in zip(true_labels, pred_labels)]
    else:
        cmaps = [None] * len(true_labels)
        labels = [str(t) for t in true_labels]
    array = mp.wrap(*[
        mp.border(
            mp.image(ddigit, colormap=cmap)
            ^ mp.center(mp.text(label), width=width)
        )
        for ddigit, label, cmap in zip(ddigits, labels, cmaps)
    ], cols=5)
    return array


def vis_metrics(
    losses: list[tuple[int, float]],
    accuracies: list[tuple[int, float]],
    total_num_steps: int,
) -> mp.plot:
    loss_plot = (
        mp.center(mp.text("train loss (cross entropy)"), width=40)
        ^ mp.border(mp.scatter(
            data=losses,
            xrange=(0, total_num_steps-1),
            yrange=(0, max(l for s, l in losses)),
            color=(1,0,1),
            width=38,
            height=11,
        ))
        ^ mp.text(f"loss: {losses[-1][1]:.3f}")
    )
    acc_plot = (
        mp.center(mp.text("test accuracy"), width=40)
        ^ mp.border(mp.scatter(
            data=accuracies,
            xrange=(0, total_num_steps-1),
            yrange=(0, 1),
            color=(0,1,0),
            width=38,
            height=11,
        ))
        ^ mp.text(f"acc: {accuracies[-1][1]:.2%}")
    )
    return loss_plot & acc_plot


# # # 
# Entry point

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
