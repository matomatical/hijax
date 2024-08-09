"""
Bitesized GPT character model for next-char-prediction based on the Holmesian
canon.

Preparation:

* any questions from homework?
* installations: nothing new.
* download data:
  ```
  curl
  https://raw.githubusercontent.com/matomatical/bitesizedGPT/main/data/sherlock-ascii.txt -o sherlock.txt
  ```

Workshop plan:

* TODO

Challenge:

* TODO
"""

from jaxtyping import Array, Float, UInt8 as Byte, PRNGKeyArray

import jax
import jax.numpy as jnp
import einops
import optax
import equinox as eqx

import tqdm
import mattplotlib


# # # 
# Architecture


class ByteSequenceModel(eqx.Module):
    decode_transformer: DecodeTransformer
    max_context_length: int

    def __init__(
        self, 
        key: PRNGKey,
        max_context_length: int,
        embed_size: int,
        mlp_size: int,
        num_heads: int,
        num_layers: int,
    ):
        self.max_context_length = max_context_length
        self.decode_transformer = DecodeTransformer(
            key=key,
            alphabet_size=128,
            max_context_length=max_context_length,
            embed_size=embed_size,
            mlp_size=mlp_size,
            num_heads=num_heads,
            num_blocks=num_blocks,
        )

    def forward(
        self,
        byte_array: Byte[Array, "t"],
    ) -> Float[Array, "t 128"]:
        tokens_one_hot = jnp.ones((128,128))[byte_array]
        prob_next_tokens = self.decode_transformer(tokens_one_hot)
        return prob_next_tokens

    def forward_batch(
        self,
        byte_arrays: Byte[Array, "b t"],
    ) -> Float[Array, "b t 128"]:
        return jax.vmap(self.forward)(byte_arrays)


    # def complete(self, prompt, max_bytes):
    #     v = prompt[:]
    #     while len(v) < max_bytes:
    #         v_ = v[None, max(0, len(v)-self.max_context_length):]
    #         last_logits = self(v_)[0, -1, :]
    #         probs = fn.softmax(last_logits, dim=0)
    #         b = torch.multinomial(probs, num_samples=1)
    #         v = torch.cat((v, b))
    #     return v


class LayerNorm(eqx.Module):
    learned_loc: Float[Array, "size"]
    learned_scale: Float[Array, "size"]

    def __init__(self, size: int):
        self.learned_loc = jnp.zeros(size)
        self.learned_scale = jnp.ones(d)

    def __call__(self, x: Float[Array, "size"]) -> Float[Array, "size"]:
        x_norm = (x - jnp.mean(x)) * jax.lax.rsqrt(jnp.var(x) + 1e-5)
        return x_norm * self.learned_scale + self.learned_mean


class DecodeTransformer(eqx.Module):
    token_embedding: Float[Array, "alphabet_size embed_size"]
    postn_embedding: Float[Array, "max_context_len embed_size"]
    blocks: list[MultiHeaderCausalSelfAttentionBlock]
    unembedding_ln: LayerNorm
    unembedding: Float[Array, "embed_size alphabet_size"]

    def __init__(
        self,
        key: PRNGKey,
        alphabet_size: int,
        max_context_length: int,
        embed_size: int,
        mlp_size: int,
        num_heads: int,
        num_blocks: int,
    ):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        bound_embed = jax.lax.rsqrt(alphabet_size)
        self.token_embedding = jax.random.uniform(
            key=k1,
            shape=(alphabet_size, embed_size),
            minval=-bound_embed,
            maxval=+bound_embed,
        )
        self.postn_embedding = jax.random.uniform(
            key=k2,
            shape=(max_context_length, embed_size),
            minval=-bound_embed,
            maxval=+bound_embed,
        )
        self.blocks = [
            MultiHeadedCausalSelfAttentionTransformerBlock(
                key=k_block,
                embed_size=embed_size,
                mlp_size=mlp_size,
                max_context_length=max_context_length,
                num_heads=num_heads,
            )
            for k_block in jax.random.split(k3, num_blocks)
        ]
        self.unembedding_ln = LayerNorm(embed_size)
        bound_unembed = jax.lax.rsqrt(embed_size)
        self.unembedding = jax.random.uniform(
            key=k4,
            shape=(embed_size, alphabet_size),
            minval=-bound_unembed,
            maxval=+bound_unembed,
        )

    def __call__(
        self,
        tokens: Float[Array, "t alphabet_size"],
    ) -> Float[Array, "t alphabet_size"]:
        t, _v = tokens.shape
        T_max, _C = self.postn_embedding.shape
        if t > T_max:
            raise ValueError(f"too many tokens! {t} > {T_max}")

        # semantic and positional token embeddings
        x_semantics = tokens @ self.token_embedding tokens  # t v @ v C -> t C
        x_positions = self.postn_embedding[:T, :]           #   T_max C -> t C
        x = x_semantics + x_positions                       # t C + t C -> t C

        # apply the num_blocks attention blocks in sequence
        for block in self.blocks:
            x = x + block(x)                                # t C + t C -> t C

        # unembedding: transform back to predicted next token probs
        logits = self.unembedding(x)                        # t C @ C v -> t v
        probs = jax.nn.softmax(y, axis=-1)
        return probs


class MultiHeadedCausalSelfAttentionBlock(eqx.Module):
    attention: MultiHeadedCausalSelfAttention
    compute: MLP
    pre_attn_ln: LayerNorm
    pre_mlp_ln: LayerNorm

    def __init__(
        self,
        key: PRNGKeyArray,
        embed_size: int,
        mlp_size: int,
        max_context_length: int,
        num_heads: int,
    ):
        k1, k2 = jax.random.split(key)
        # attention
        self.pre_attn_ln = LayerNorm(size=embed_size)
        self.attention = MultiHeadedCausalSelfAttention(
            key=k1,
            embed_size=embed_size,
            max_context_length=max_context_length,
            num_heads=num_heads,
        )
        # compute
        self.pre_mlp_ln = LayerNorm(size=embed_size)
        self.compute = MLP(
            key=k2,
            num_inputs=embed_size,
            num_hidden=mlp_size,
            num_outputs=embed_size,
        )

    def __call__(
        self,
        x: Float[Array, "t embed_size"],
    ) -> Float[Array, "t embed_size"]:
        x = x + self.attention(self.pre_attn_ln(x))
        x = x + jax.vmap(self.compute)(jax.vmap(self.pre_mlp_ln)(x))
        return x


class MLP(eqx.Module):
    weights1: Float[Array, "num_inputs num_hidden"] 
    biases1: Float[Array, "num_hidden"]
    weights2: Float[Array, "num_hidden num_outputs"]
    biases2: Float[Array, "num_outputs"]

    def __init__(
        self,
        key: PRNGKeyArray,
        num_inputs: int,
        num_hidden: int,
        num_outputs: int,
    ):
        k1, k2 = jax.random.split(key)
        # layer 1
        bound_layer1 = jax.lax.rsqrt(num_inputs)
        self.weights1 = jax.random.uniform(
            key=k1,
            shape=(num_inputs, num_hidden),
            minval=-bound_layer1,
            maxval=+bound_layer1,
        )
        self.biases1 = jnp.zeros(num_hidden)
        # layer 2
        bound_layer2 = jax.lax.rsqrt(num_hidden)
        self.weights2 = jax.random.uniform(
            key=k2,
            shape=(num_hidden, num_outputs),
            minval=-bound_layer2,
            maxval=+bound_layer2,
        )
        self.biases2 = jnp.zeros(num_outputs)

    def __call__(
        self,
        x: Float[Array, "num_inputs"],
    ) -> Float[Array, "num_outputs"]:
        x = x @ self.weights1 + self.biases1
        x = jax.nn.relu(x)
        x = x @ self.weights2 + self.biases2
        return x


class MultiHeadedCausalSelfAttention(eqx.Module):
    attention_maps: Float[Array, "embed_size 3*embed_size"]
    num_heads: int
    head_size: int

    def __init__(
        self,
        key: PRNGKeyArray,
        embed_size: int,
        max_context_length: int,
        num_heads: int,
    ):
        # validate dimensions
        if embed_size % num_heads:
            raise ValueError("num_heads must divide embed_size")
        self.num_heads = num_heads
        self.head_size = embed_size // num_heads

        # batched key/query/value projections
        bound_attn = jax.lax.rsqrt(embed_size)
        self.attention_maps = jax.random.uniform(
            key=key,
            shape=(embed_size, 3*embed_size),
            minval=-bound_attn,
            maxval=+bound_attn,
        )
        
        # TODO: I'M UP TO HERE
        # precompute causal mask
        mask_shape = (max_context_length, max_context_length)
        causal_mask = torch.log(torch.tril(torch.ones(mask_shape)))
        self.register_buffer('causal_mask', causal_mask)
        # precompute attention normalisation factor
        self.attention_scale = self.head_size ** 0.5

    def __call__(self, x):
        # unpack dimensions
        B, T, C = x.size()  # batch size, num_tokens, embed_size
        H = self.num_heads  # num_heads
        c = self.head_size  # head size

        # perform Q, K, V transforms, all at once
        Q, K, V = (self.attention(x)    # B T C @ C 3C  -> B T 3C
                .view(B, T, H, 3*c)     #               -> B T H 3c
                .transpose(-2, -3)      #               -> B H T 3c
                .split(c, dim=-1)       #               -> (B H T c) * 3
            )
        # now Q, K, V are each of shape (B, H, T, c)

        # compute affinities, scaled and with causal mask
        A = Q @ K.transpose(-2, -1)     # B H T c @ B H c T -> B H T T
        A = A / self.attention_scale    # B H T T / . . . T -> B H T T
        A = A + self.causal_mask[:T,:T] # B H T T + . . T T -> B H T T

        # convert affinities to mixing weights and mix value vectors
        p = fn.softmax(A, dim=-1)   # B H T T -> B H T T(sum to 1)
        y = p @ V                   # B H T T @ B H T c -> B H T c

        # recombine / concatenate heads into new embedding
        y = (y                      #    B H T c
                .transpose(-3, -2)  # -> B T H c
                .contiguous()       # -> (make underlying memory match view)
                .view(B, T, C)      # -> B T C
             )

        return y


# # # 
# Helper functions


# def next_byte_cross_entropy_loss(bytes_, next_byte_logits):
#     B, T, V = next_byte_logits.shape
#     next_bytes = bytes_[:, 1:].reshape(B*(T-1))
#     next_byte_logits = next_byte_logits[:, :-1, :].reshape(B*(T-1), V)
#     return fn.cross_entropy(next_byte_logits, next_bytes)


# def complete(model, prompt, max_bytes):
#     b = str2bytevec(prompt)
#     c = model.complete(b, max_bytes=max_bytes)
#     s = bytevec2str(c)
#     return s


# # # 
# Training loop


def main(
    seed: int = 221,
):
    key = jax.random.key(seed)
    

    print("loading byte corpus...")
    with open("sherlock.txt") as file:
        data = str_to_array(file.read())
    data_train = data[:3_000_000]
    data_test = data[3_000_000:]

    print(data.shape)
    print(repr(array_to_str(data_test[:32])))
    

    print("initialising model...")
    model = BiteSizedGenerativePretrainedTransformer(
        max_context_length=128,
        embed_size=64,
        mlp_size=64,
        num_heads=4,
        num_blocks=4,
    )


def str_to_array(s: str) -> Byte[Array, "len(s)"]:
    return jnp.array([ord(c) for c in s], dtype=jnp.uint8)


def array_to_str(a: Byte[Array, "len(s)"]) -> str:
    return "".join(chr(i) for i in a)



def train():
    
    # initialising training stuff
    num_training_steps = 50000
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    print("training model...")
    for steps in tqdm.trange(num_training_steps):
        batch = data.get_training_batch(seq_length=128, batch_size=32)
#     def _get_batch(self, data, seq_length, batch_size):
#         idx_start = torch.randint(len(data)-seq_length, (batch_size,))
#         idx = idx_start.view(-1, 1) + torch.arange(seq_length)
#         return data[idx]
        logits = model(batch)
        loss = next_byte_cross_entropy_loss(batch, logits)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # evaluate periodically
        if steps % 100 == 0:
            tqdm.tqdm.write(f"eval at step {steps:>6d}")
            model.eval()
            # batch loss
            tqdm.tqdm.write(f"  training loss: {loss.item():>6.3f}")
            # test loss
            eval_batch = data.get_testing_batch(seq_length=128, batch_size=256)
            with torch.no_grad():
                logits = model(eval_batch)
            loss = next_byte_cross_entropy_loss(eval_batch, logits)
            tqdm.tqdm.write(f"  testing loss:  {loss.item():>6.3f}")
            # prompt
            with torch.no_grad():
                ctn = complete(model, '"Elementary, my dear', max_bytes=32)
            tqdm.tqdm.write(f"  continuation:  {ctn!r}")
            model.train()

    print("done!")
    print("generating passage from model...")
    model.to('cpu')
    with torch.no_grad():
        ctn = complete(model, '"Elementary, my dear', max_bytes=256)
    for c in ctn:
        print(c, end="", flush=True)
        time.sleep(0.06125)


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
