# Sudoku-solver
This repository contains a small sudoku solving algorithm, which uses a home-cooked evolutionary algorithm 👨‍🍳.

_Note: This is by no means a fast implementation. If you're looking for a something performant, check out [this implementation in Rust](https://github.com/LukasHedegaard/sudoku-solver-rust), which uses backtracking instead of evolution 🚀._

## Usage
```python
import numpy as np
from evolutionary_search import evolutionary_search

# Zeros correspond to empty fields
board = np.array(
    [
        [6, 3, 0, 8, 0, 2, 0, 0, 9],
        [4, 0, 0, 0, 0, 6, 0, 7, 0],
        [0, 0, 2, 4, 0, 0, 8, 0, 3],
        [0, 0, 0, 0, 0, 8, 9, 3, 0],
        [0, 7, 9, 0, 6, 5, 0, 4, 1],
        [2, 0, 3, 7, 0, 0, 0, 0, 0],
        [1, 0, 6, 0, 0, 0, 0, 0, 0],
        [9, 5, 0, 6, 0, 3, 0, 2, 4],
        [0, 0, 0, 0, 4, 9, 0, 1, 8],
    ]
)

solution = evolutionary_search(
    board=board,
    population_size=1000,
    max_iters=1000,
    elitism=0.1,
    randomism=0.1,
    mutation_prob=1e-4,
    patience=100,
    temperature=3,
    parent_replacement=False,
)
print("Solution:")
print(solution)
```
Output:
```bash
Best = 0, Avg = -9.7:   4%|▎   | 37/1000 [00:09<04:08,  3.88it/s]   
Stopping criterion reached 💪
Solution:
array([[6, 3, 1, 8, 7, 2, 4, 5, 9],
       [4, 8, 5, 9, 3, 6, 1, 7, 2],
       [7, 9, 2, 4, 5, 1, 8, 6, 3],
       [5, 6, 4, 1, 2, 8, 9, 3, 7],
       [8, 7, 9, 3, 6, 5, 2, 4, 1],
       [2, 1, 3, 7, 9, 4, 5, 8, 6],
       [1, 4, 6, 2, 8, 7, 3, 9, 5],
       [9, 5, 8, 6, 1, 3, 7, 2, 4],
       [3, 2, 7, 5, 4, 9, 6, 1, 8]])
```
