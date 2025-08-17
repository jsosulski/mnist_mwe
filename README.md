# MWE Burn MNIST

This repo implements the MNIST example from the burn book.

## Issue

Running the training with WGPU results in a reasonably well trained network with >90% validation accuracy.

```bash
$ cargo run --release --features wgpu
...
| Split | Metric   | Min.     | Epoch    | Max.     | Epoch    |
|-------|----------|----------|----------|----------|----------|
| Train | Accuracy | 82.317   | 1        | 82.317   | 1        |
| Train | Loss     | 0.615    | 1        | 0.615    | 1        |
| Valid | Accuracy | 92.300   | 1        | 92.300   | 1        |
| Valid | Loss     | 0.252    | 1        | 0.252    | 1        |
```

However, training on CPU yields a validation accuracy of ~10-15%, which is basically chance level.

```bash
$ cargo run --release
...
| Split | Metric   | Min.     | Epoch    | Max.     | Epoch    |
|-------|----------|----------|----------|----------|----------|
| Train | Accuracy | 73.732   | 1        | 73.732   | 1        |
| Train | Loss     | 1.006    | 1        | 1.006    | 1        |
| Valid | Accuracy | 8.470    | 1        | 8.470    | 1        |
| Valid | Loss     | 3.637    | 1        | 3.637    | 1        |
```
