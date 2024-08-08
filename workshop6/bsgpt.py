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

from jaxtyping import Array, Float, UInt8, PRNGKeyArray

import jax
import jax.numpy as jnp
import einops
import optax

import tqdm
import mattplotlib


# # # 
# Architecture


# class ByteTransformer(nn.Module):
#     def __init__(
#         self,
#         max_context_length,
#         embed_size,
#         mlp_size,
#         num_heads,
#         num_layers,
#         device=None,
#     ):
#         super().__init__()
#         self.max_context_length = max_context_length
#         self.decode_transformer = DecodeTransformer(
#             max_context_length=max_context_length,
#             alphabet_size=128,
#             embed_size=embed_size,
#             mlp_size=mlp_size,
#             num_heads=num_heads,
#             num_layers=num_layers,
#             device=device,
#         )
#         self.register_buffer('onehot', torch.eye(128, device=device))
#         self.device = device
# 
#     def to(self, device):
#         super().to(device)
#         self.device = device
# 
#     def forward(self, bytes_):
#         tokens = self.onehot[bytes_]
#         logits = self.decode_transformer(tokens)
#         return logits
# 
#     def complete(self, prompt, max_bytes):
#         v = prompt[:]
#         while len(v) < max_bytes:
#             v_ = v[None, max(0, len(v)-self.max_context_length):]
#             last_logits = self(v_)[0, -1, :]
#             probs = fn.softmax(last_logits, dim=0)
#             b = torch.multinomial(probs, num_samples=1)
#             v = torch.cat((v, b))
#         return v


# class DecodeTransformer(nn.Module):
#     def __init__(
#         self,
#         max_context_length,
#         alphabet_size,
#         embed_size,
#         mlp_size,
#         num_heads,
#         num_layers,
#         device=None,
#     ):
#         super().__init__()
#         self.token_embedding = nn.Linear(
#             in_features=alphabet_size,
#             out_features=embed_size,
#             bias=False,
#             device=device,
#         )
#         self.postn_embedding = nn.Linear(
#             in_features=max_context_length,
#             out_features=embed_size,
#             bias=False,
#             device=device,
#         )
#         self.blocks = nn.ModuleList([
#             MultiHeadedCausalSelfAttentionTransformerBlock(
#                 embed_size=embed_size,
#                 mlp_size=mlp_size,
#                 max_context_length=max_context_length,
#                 num_heads=num_heads,
#                 device=device,
#             )
#             for _ in range(num_layers)
#         ])
#         # unembedding
#         self.unembedding = nn.Sequential(
#             nn.LayerNorm(
#                 normalized_shape=embed_size,
#                 device=device,
#             ),
#             nn.Linear(
#                 in_features=embed_size,
#                 out_features=alphabet_size,
#                 device=device,
#             ),
#         )
#         self.max_context_length = max_context_length
# 
#     def forward(self, toks):
#         _B, T, _V = toks.shape
#         T_max = self.max_context_length
#         if T > T_max:
#             raise ValueError(f"too many tokens! {T} > {T_max}")
# 
#         # semantic and positional token embeddings
#         x_positions = self.postn_embedding.weight.T[:T, :] # Tmax C ->   T C
#         x_semantics = self.token_embedding(toks)    # B T V @ . V C -> B T C
#         x = x_semantics + x_positions               # B T C + . T C -> B T C
# 
#         # apply the num_layers layers / attention blocks in sequence
#         for block in self.blocks:
#             x = x + block(x)                        # B T C + B T C -> B T C
# 
#         # unembedding: transform back to predicted next tokens
#         y = self.unembedding(x)                     # B T C @ . C V -> B T V
#         
#         return y
#         # TODO: optimise
#         # during training,  we only care about y[:, :-1, :]...
#         # during inference, we only care about y[:, -1:, :]...


# class MultiHeadedCausalSelfAttentionTransformerBlock(nn.Module):
#     def __init__(
#         self,
#         embed_size,
#         mlp_size,
#         max_context_length,
#         num_heads,
#         device=None,
#     ):
#         super().__init__()
#         self.attention = MultiHeadedCausalSelfAttention(
#             embed_size=embed_size,
#             max_context_length=max_context_length,
#             num_heads=num_heads,
#             device=device,
#         )
#         self.compute = nn.Sequential(
#             nn.Linear(embed_size, mlp_size, device=device),
#             nn.ReLU(),
#             nn.Linear(mlp_size, embed_size, device=device),
#         )
#         self.layer_norms = nn.ModuleList([
#             nn.LayerNorm(normalized_shape=embed_size, device=device)
#             for _ in ('before-attention', 'before-compute')
#         ])
# 
#     def forward(self, x):
#         # B, T, C = x.shape
#         x = x + self.attention(self.layer_norms[0](x))
#         x = x + self.compute(self.layer_norms[1](x))
#         return x


# class MultiHeadedCausalSelfAttention(nn.Module):
#     def __init__(
#         self,
#         embed_size,
#         max_context_length,
#         num_heads,
#         device=None,
#     ):
#         super().__init__()
#         # validate dimensions
#         if embed_size % num_heads:
#             raise ValueError("num_heads must divide embed_size")
#         self.num_heads = num_heads
#         self.head_size = embed_size // num_heads
#         # batched key/query/value projections
#         self.attention = nn.Linear(
#             in_features=embed_size,
#             out_features=3*embed_size,
#             bias=False,
#             device=device,
#         )
#         # precompute causal mask
#         mask_shape = (max_context_length, max_context_length)
#         causal_mask = torch.log(torch.tril(torch.ones(mask_shape)))
#         self.register_buffer('causal_mask', causal_mask.to(device))
#         # precompute attention normalisation factor
#         self.attention_scale = self.head_size ** 0.5
# 
#     def forward(self, x):
#         # unpack dimensions
#         B, T, C = x.size()  # batch size, num_tokens, embed_size
#         H = self.num_heads  # num_heads
#         c = self.head_size  # head size
# 
#         # perform Q, K, V transforms, all at once
#         Q, K, V = (self.attention(x)    # B T C @ C 3C  -> B T 3C
#                 .view(B, T, H, 3*c)     #               -> B T H 3c
#                 .transpose(-2, -3)      #               -> B H T 3c
#                 .split(c, dim=-1)       #               -> (B H T c) * 3
#             )
#         # now Q, K, V are each of shape (B, H, T, c)
# 
#         # compute affinities, scaled and with causal mask
#         A = Q @ K.transpose(-2, -1)     # B H T c @ B H c T -> B H T T
#         A = A / self.attention_scale    # B H T T / . . . T -> B H T T
#         A = A + self.causal_mask[:T,:T] # B H T T + . . T T -> B H T T
# 
#         # convert affinities to mixing weights and mix value vectors
#         p = fn.softmax(A, dim=-1)   # B H T T -> B H T T(sum to 1)
#         y = p @ V                   # B H T T @ B H T c -> B H T c
# 
#         # recombine / concatenate heads into new embedding
#         y = (y                      #    B H T c
#                 .transpose(-3, -2)  # -> B T H c
#                 .contiguous()       # -> (make underlying memory match view)
#                 .view(B, T, C)      # -> B T C
#              )
# 
#         return y


# # # 
# Helper functions


# def next_byte_cross_entropy_loss(bytes_, next_byte_logits):
#     B, T, V = next_byte_logits.shape
#     next_bytes = bytes_[:, 1:].reshape(B*(T-1))
#     next_byte_logits = next_byte_logits[:, :-1, :].reshape(B*(T-1), V)
#     return fn.cross_entropy(next_byte_logits, next_bytes)


# def complete(model, prompt, max_bytes):
#     b = str2bytevec(prompt, device=model.device)
#     c = model.complete(b, max_bytes=max_bytes)
#     s = bytevec2str(c)
#     return s


# # # 
# Training loop


def main():
    print("loading byte corpus...")
    with open("sherlock.txt") as file:
        data = str_to_array(file.read())
    data_train = data[:3_000_000]
    data_test = data[3_000_000:]

    print(data.shape)
    print(array_to_str(data_test[:10000]))


def str_to_array(s: str) -> UInt8[Array, "len(s)"]:
    return jnp.array([ord(c) for c in s], dtype=jnp.uint8)


def array_to_str(a: UInt8[Array, "len(s)"]) -> str:
    return "".join(chr(i) for i in a)





def train(device):
    print("initialising model...")
    model = ByteTransformer(
        max_context_length=128,
        embed_size=32,
        mlp_size=64,
        num_heads=4,
        num_layers=4,
        device=device,
    )
    model.train()
    
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
