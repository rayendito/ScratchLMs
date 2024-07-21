# GPTnano (nanoGPT but named backwards) (but soon to be renamed, i think, anyways)
some NN architectures that are implemented 'from scratch'* in PyTorch

## Implemented Architectures
1. **GPT** – Decoder only Transformers (based on Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT)) ([debugging notebook](https://colab.research.google.com/drive/1y9XAKnat5iWr3H4jsWgqDsW6frrm9_sy?usp=sharing))
2. RNN – a simple recurrent network (Elman network, based on [this wikipedia page](https://en.wikipedia.org/wiki/Recurrent_neural_network) lol)
3. Coming soon, maybe? [BackpackLM](https://arxiv.org/abs/2305.16765)
3. Coming soon, maybe? CNN


*why the air quotes? well, there are many opinions on what 'from scratch' means and in this one i mean i'm implementing them as long as i'm not using the prebuilt pytorch modules like nn.RNN. one can theoretically go as extreme as making stuff from sand, working their way up to a working NN through making their own silicone, transistors, processors, OS, and all that jazz, kinda like [the toaster project](https://www.thomasthwaites.com/the-toaster-project/). but for the time being, we're gonna start pretty high up in the layers of abstraction :)
