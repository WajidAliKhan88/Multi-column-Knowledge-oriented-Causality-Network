# MCKN
Toward Multi-column Knowledge-oriented Neural Network for Web Corpus Causality Mining

## Published. [Public Version](Nill)

## Prerequisites

- Pytorch >= 0.4
- NLTK
- gensim
- Sklearn


## Preprocessing

Preprocess the trainset:

```
python torch_run.py run --prepare=True --build=True
```

## Train

```
python torch_run.py run --train=True
```

## Test

```
python torch_run.py run  --evaluate=True
```
## Dataset

[altlex](https://github.com/chridey/altlex)


## More details will be added with the passage of time...