import numpy as np
from tqdm import tqdm
import random
from copy import deepcopy as copy

numerals = set(range(1, 10))
block_permutation = np.arange(9).reshape((3, 3)).T.flatten()


def sel(l, inds):
    return [l[i] for i in inds]


def shuffled(x):
    return sorted(x, key=lambda k: random.random())


def get_blocks(board):
    return [
        board[i * 3 : i * 3 + 3, j * 3 : j * 3 + 3].flatten()
        for i in range(3)
        for j in range(3)
    ]


def initial_state(board):
    # Init sub-blocks to be valid
    return [shuffled(numerals - set(b)) for b in get_blocks(board)]


def unflatten_block(b):
    return b.reshape((3, 3))


def fill_block(block, block_state):
    block[block == 0] = block_state
    return block


def unflatten_blocks(blocks):
    return np.block(
        [[blocks[3 * j + i].reshape((3, 3)) for i in range(3)] for j in range(3)]
    )


def fill_board(board, state):
    blocks = get_blocks(board)
    return unflatten_blocks([fill_block(b, s) for b, s in zip(blocks, state)])


def transpose_state(state):
    return [state[i] for i in block_permutation]


def get_groups(b):
    cols = [b[:, i] for i in range(9)]
    rows = [b[i, :] for i in range(9)]
    blocks = [
        b[i * 3 : i * 3 + 3, j * 3 : j * 3 + 3].flatten()
        for i in range(3)
        for j in range(3)
    ]
    return cols + rows + blocks


def group_loss(g):
    return len(list(g)) - len(set(g))


def loss(b):
    # Add penalty for each repeat elem in row, col and block
    return sum([group_loss(g) for g in get_groups(b)])


def crossover(s1, s2, b):
    assert len(s1) == len(s2)
    do_transpose = np.random.uniform() > 0.5
    if do_transpose:
        s1 = transpose_state(s1)
        s2 = transpose_state(s2)

    if np.random.uniform() < 0.05:
        i = np.random.randint(low=0, high=9, size=1).item()
        c1, c2 = swap2(s1, s2, i)

    if np.random.uniform() < 0.5:
        i = np.random.randint(low=0, high=9, size=1).item()
        c1, c2 = swap2(s1, s2, i)
    else:
        i = np.random.randint(low=1, high=3, size=1).item()
        c1 = s1[: i * 3] + s2[i * 3 :]
        c2 = s2[: i * 3] + s1[i * 3 :]

    if do_transpose:
        c1 = transpose_state(c1)
        c2 = transpose_state(c2)

    return c1, c2


def round2(x):
    return int(x // 2) * 2


def softmax(x, temperature=1):
    """Compute softmax values for each sets of scores in x."""
    m = np.max(x)
    e_x = np.exp((x - m) / temperature)
    return e_x / e_x.sum(axis=0)


def initial_population(b, size=20):
    return [initial_state(b) for _ in range(size)]


def population_fitness(pop, board):
    fitness = [-loss(fill_board(board, s)) for s in pop]
    return fitness


def select_fittest(pop, fitness, num):
    inds = list(np.array(fitness).argsort()[::-1][:num])
    return sel(pop, inds), sel(fitness, inds)


def select_random_fittest(pop, fitness, num, temperature=3, replace=True):
    fitness = softmax(fitness, temperature=temperature)  # hparam
    inds = np.random.choice(np.arange(len(pop)), size=num, replace=replace, p=fitness)
    return sel(pop, inds)


def recombine_population(pop, b):
    assert len(pop) > 0
    assert len(pop) % 2 == 0
    pop = shuffled(pop)
    half = len(pop) // 2
    children = [
        crossover(copy(p1), copy(p2), b) for p1, p2 in zip(pop[:half], pop[half:])
    ]
    children = list(sum(children, ()))  # flatten tuples
    return children


def swap(lst, i, j):
    get = lst[i], lst[j]
    lst[j], lst[i] = get
    return lst


def swap2(l1, l2, i):
    mem1, mem2 = l1[i], l2[i]
    l1[i], l2[i] = mem2, mem1
    return l1, l2


def mutate(block):
    i = np.random.randint(low=0, high=len(block), size=2)
    return swap(block, i[0], i[1])


def mutate_population(pop, prob=1e-4):
    return [
        [block if np.random.uniform() > prob else mutate(block) for block in blocks]
        for blocks in pop
    ]


def unpack_winner(w, b):
    s, l = w
    b_ful = fill_board(b, s[0])
    return {"Solution": b_ful, "Fitness": l}


def evolutionary_search(
    board,
    population_size=100,
    max_iters=10,
    elitism=0.1,
    randomism=0.1,
    mutation_prob=1e-2,
    stopping_fitness=0,
    patience=50,
    temperature=2,
    parent_replacement=True,
):
    alltime_best = -np.inf
    gens_since_best = 0
    pop = initial_population(board, population_size)
    with tqdm(total=max_iters) as pbar:
        for i in range(max_iters):
            fitness = population_fitness(pop, board)

            # Keep track of best
            best_solution, best_fitness = select_fittest(pop, fitness, 1)
            best_fitness = best_fitness[0]
            pbar.set_description(f"Best = {best_fitness}, Avg = {np.mean(fitness):.1f}")

            if best_fitness >= stopping_fitness:
                print("Stopping criterion reached 💪")
                return unpack_winner((best_solution, best_fitness), board)

            if best_fitness > alltime_best:
                alltime_best = best_fitness
                gens_since_best = 0
            else:
                gens_since_best += 1
            if gens_since_best >= patience:
                alltime_best = -np.inf
                gens_since_best = 0
                pop = initial_population(board, population_size)
                print("Restarting")
                continue

            num_elite = round2(elitism * population_size)
            elite, elite_fitness = select_fittest(pop, fitness, num_elite)

            num_rands = round2(randomism * population_size)
            rands = initial_population(board, num_rands)

            num_parents = round2(len(pop) - num_elite - num_rands)
            fittest_parents = select_random_fittest(
                pop, fitness, num_parents, temperature, parent_replacement
            )
            parents = fittest_parents + rands
            children = recombine_population(parents, board)
            children = mutate_population(children, mutation_prob)

            pop = elite + children
            pbar.update(1)
    print("Stopping criterion not reached 😭")
    return unpack_winner(select_fittest(pop, fitness, 1), board)
