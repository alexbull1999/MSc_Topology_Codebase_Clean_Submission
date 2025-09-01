import torch
import geoopt
import ripser
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.datasets import make_blobs

# Test hyperbolic operations
def test_hyperbolic():
    ball = geoopt.PoincareBall()
    x = torch.randn(10, 50) * 0.1
    x_hyp = ball.expmap0(x)
    distances = ball.dist(x_hyp[0:1], x_hyp[1:])
    print(f"Hyperbolic test passed: {distances[:3]}")

# Test TDA
def test_tda():
    X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
    diagrams = ripser.ripser(X)['dgms']
    print(f"TDA test passed: Found {len(diagrams[1])} 1D features")

# Test transformers
def test_transformers():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer("Hello world", return_tensors="pt")
    outputs = model(**inputs)
    print(f"BERT test passed: {outputs.last_hidden_state.shape}")

if __name__ == "__main__":
    test_hyperbolic()
    test_tda()
    test_transformers()