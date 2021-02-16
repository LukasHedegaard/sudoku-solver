import numpy as np
from evolutionary_search import evolutionary_search

# Boards are encoded with '0' correspondonig to a missing field
easy_board = np.array(
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

medium_board = np.array(
    [
        [0, 5, 0, 7, 6, 2, 8, 0, 0],
        [0, 0, 0, 0, 0, 5, 0, 6, 0],
        [8, 0, 0, 4, 1, 0, 0, 0, 7],
        [0, 6, 0, 0, 0, 0, 4, 0, 8],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, 8, 0, 0, 0, 0, 7, 0],
        [5, 0, 0, 0, 8, 7, 0, 0, 4],
        [0, 9, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 7, 6, 5, 3, 0, 9, 0],
    ]
)

hard_board = np.array(
    [
        [0, 0, 8, 0, 1, 0, 9, 3, 0],
        [0, 0, 4, 0, 0, 5, 2, 0, 0],
        [0, 0, 0, 8, 0, 0, 0, 6, 0],
        [3, 0, 2, 0, 4, 0, 1, 0, 0],
        [5, 0, 0, 0, 0, 0, 0, 0, 3],
        [0, 0, 9, 0, 3, 0, 7, 0, 2],
        [0, 2, 0, 0, 0, 9, 0, 0, 0],
        [0, 0, 1, 4, 0, 0, 3, 0, 0],
        [0, 6, 3, 0, 8, 0, 5, 0, 0],
    ]
)

# ≈ 10 seconds
easy_solution = evolutionary_search(
    board=easy_board,
    population_size=1000,
    max_iters=1000,
    elitism=0.1,
    randomism=0.1,
    mutation_prob=1e-4,
    patience=100,
    temperature=3,
    parent_replacement=False,
)
print(easy_solution)

# ≈ 10 minutes
medium_solution = evolutionary_search(
    board=medium_board,
    population_size=4000,
    max_iters=1000,
    elitism=0.1,
    randomism=0.1,
    mutation_prob=1e-3,
    patience=300,
    temperature=3,
    parent_replacement=False,
)
print(medium_solution)

# ≈ 30 minutes
hard_solution = evolutionary_search(
    board=hard_board,
    population_size=5000,
    max_iters=2000,
    elitism=0.1,
    randomism=0.15,
    mutation_prob=5e-2,
    patience=1000,
    temperature=3,
    parent_replacement=False,
)
print(medium_solution)
