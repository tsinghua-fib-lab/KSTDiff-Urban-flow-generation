# Urban flow generation

Official implementation of "Towards Generative Modeling of Urban Flow through Knowledge-enhanced Denoising Diffusion"(SIGSPATIAL'23).
NYC dataset is included.

# Usage

Pretrain to get KG embeddings:
```
bash pretrain.sh
```

Train diffusion model:

```
bash train.sh
```

Generate urban flow:

```
bash sample.sh
```

Evaluate generated flow:

```
python evaluate.py
```

# Reference
```

```
