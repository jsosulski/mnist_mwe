# MWE Burn MNIST

This repo implements the MNIST example from the burn book.
I tried it and noticed an issue with convergence between wgpu/ndarray backends.
However, the issue is now resolved.

## Results with latest burn version (tested with =0.22.0-pre.4)

Output from WGPU:

```bash
$ cargo run --release --features wgpu
...
| Split | Metric   | Min.     | Epoch    | Max.     | Epoch    |
|-------|----------|----------|----------|----------|----------|
| Train | Accuracy | 82.410   | 1        | 82.410   | 1        |
| Train | Loss     | 0.619    | 1        | 0.619    | 1        |
| Valid | Accuracy | 93.170   | 1        | 93.170   | 1        |
| Valid | Loss     | 0.234    | 1        | 0.234    | 1        |

Total runtime: 32.65s
```

Output from ndarray:

```bash
$ cargo run --release
...
| Split | Metric   | Min.     | Epoch    | Max.     | Epoch    |
|-------|----------|----------|----------|----------|----------|
| Train | Accuracy | 83.315   | 1        | 83.315   | 1        |
| Train | Loss     | 0.589    | 1        | 0.589    | 1        |
| Valid | Accuracy | 93.280   | 1        | 93.280   | 1        |
| Valid | Loss     | 0.233    | 1        | 0.233    | 1        |

Total runtime: 364.01s
```

Output from CubeCL CPU:

```bash
$ cargo run --release --features cubecl
...
| Split | Metric   | Min.     | Epoch    | Max.     | Epoch    |
|-------|----------|----------|----------|----------|----------|
| Train | Accuracy | 82.915   | 1        | 82.915   | 1        |
| Train | Loss     | 0.595    | 1        | 0.595    | 1        |
| Valid | Accuracy | 93.150   | 1        | 93.150   | 1        |
| Valid | Loss     | 0.230    | 1        | 0.230    | 1        |

Total runtime: 226.91s
```

Output from burn-flex:

```bash
$ cargo run --release --features flex
...
| Split | Metric   | Min.     | Epoch    | Max.     | Epoch    |
|-------|----------|----------|----------|----------|----------|
| Train | Accuracy | 83.667   | 1        | 83.667   | 1        |
| Train | Loss     | 0.586    | 1        | 0.586    | 1        |
| Valid | Accuracy | 93.360   | 1        | 93.360   | 1        |
| Valid | Loss     | 0.228    | 1        | 0.228    | 1        |

Total runtime: 32.67s
```

## OUTDATED: Old output with issue present (was present on at least 0.18.0 but I also tested some newer versions <0.22.0-pre.4)

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

However, training on CPU with ndarray yields a validation accuracy of ~10-15%, which is basically chance level.

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
