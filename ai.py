import os
import sys
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import pandas as pd
import matplotlib.pyplot as plt

# Suppress TensorFlow INFO logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1) Loader for OR-Library flowshop files ---
def load_orlib_flowshop(file_path):
    with open(file_path, 'r') as f:
        tokens = f.read().split()
    nums = [int(t) for t in tokens if t.lstrip('-').isdigit()]
    instances = []
    i, L = 0, len(nums)
    while i + 2 <= L:
        n_jobs, n_macs = nums[i], nums[i+1]
        i += 2
        total = n_jobs * n_macs
        if i + total > L:
            break
        block = nums[i:i+total]
        i += total
        mat = np.array(block, dtype=int).reshape((n_jobs, n_macs))
        instances.append(mat)
    return instances

# --- 2) Genetic Algorithm for PFSP ---
class GeneticAlgorithm:
    def __init__(self, pt, pop_size=50, cx_rate=0.8, mut_rate=0.2):
        self.pt        = pt
        self.n         = pt.shape[0]
        self.pop_size  = pop_size
        self.cx_rate   = cx_rate
        self.mut_rate  = mut_rate
        self.population = [random.sample(range(self.n), self.n) for _ in range(pop_size)]

    def fitness(self, indiv):
        m = self.pt.shape[1]
        comp = np.zeros((len(indiv)+1, m+1))
        for idx, job in enumerate(indiv, start=1):
            for mach in range(1, m+1):
                comp[idx, mach] = max(comp[idx-1, mach], comp[idx, mach-1]) + self.pt[job, mach-1]
        return comp[-1, -1]

    def evaluate(self):
        return [self.fitness(ind) for ind in self.population]

    def select(self, fits):
        selected = []
        for _ in range(self.pop_size):
            i, j = random.sample(range(self.pop_size), 2)
            winner = self.population[i] if fits[i] < fits[j] else self.population[j]
            selected.append(winner.copy())
        return selected

    def crossover(self, p1, p2):
        a, b = sorted(random.sample(range(self.n), 2))
        child = [-1]*self.n
        child[a:b] = p1[a:b]
        ptr = b
        for gene in p2[b:] + p2[:b]:
            if gene not in child:
                child[ptr % self.n] = gene
                ptr += 1
        return child

    def mutate(self, indiv):
        i, j = random.sample(range(self.n), 2)
        indiv[i], indiv[j] = indiv[j], indiv[i]

    def next_gen(self, inject=None):
        fits    = self.evaluate()
        parents = self.select(fits)
        newpop  = []
        if inject:
            for seq in inject:
                if len(newpop) < self.pop_size:
                    newpop.append(seq)
        while len(newpop) < self.pop_size:
            p1, p2 = random.sample(parents, 2)
            child = self.crossover(p1, p2) if random.random() < self.cx_rate else p1.copy()
            if random.random() < self.mut_rate:
                self.mutate(child)
            newpop.append(child)
        self.population = newpop

# --- 3) GAN Definitions & Training ---
def build_generator(latent_dim, seq_len):
    inp = Input((latent_dim,))
    x = layers.Dense(128, activation='relu')(inp)
    x = layers.Dense(256, activation='relu')(x)
    out = layers.Dense(seq_len, activation='softmax')(x)
    return Model(inp, out, name='generator')

def build_discriminator(seq_len):
    inp = Input((seq_len,))
    x = layers.Dense(256, activation='relu')(inp)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return Model(inp, out, name='discriminator')

class GAN:
    def __init__(self, seq_len, latent_dim=100):
        self.gen   = build_generator(latent_dim, seq_len)
        self.disc  = build_discriminator(seq_len)
        self.opt_g = tf.keras.optimizers.Adam()
        self.opt_d = tf.keras.optimizers.Adam()
        self.loss  = tf.keras.losses.BinaryCrossentropy()

    def train(self, real_seqs, epochs=300, batch_size=32):
        valid = tf.ones((batch_size, 1))
        fake  = tf.zeros((batch_size, 1))
        for epoch in range(1, epochs+1):
            idx  = np.random.randint(0, real_seqs.shape[0], batch_size)
            real = tf.convert_to_tensor(real_seqs[idx], dtype=tf.float32)
            noise = tf.random.normal((batch_size, self.gen.input_shape[1]))
            gen_sample = self.gen(noise, training=False)
            with tf.GradientTape() as tape:
                d_loss = 0.5 * (
                    self.loss(valid, self.disc(real,    training=True)) +
                    self.loss(fake,  self.disc(gen_sample, training=True))
                )
            grads_d = tape.gradient(d_loss, self.disc.trainable_weights)
            self.opt_d.apply_gradients(zip(grads_d, self.disc.trainable_weights))
            with tf.GradientTape() as tape:
                gen2 = self.gen(noise, training=True)
                g_loss = self.loss(valid, self.disc(gen2, training=False))
            grads_g = tape.gradient(g_loss, self.gen.trainable_weights)
            self.opt_g.apply_gradients(zip(grads_g, self.gen.trainable_weights))
            if epoch % 100 == 0:
                print(f"GAN Epoch {epoch:3d}  D_loss={d_loss:.4f}  G_loss={g_loss:.4f}")

    def generate(self, n_samples):
        noise = tf.random.normal((n_samples, self.gen.input_shape[1]))
        probs = self.gen(noise, training=False).numpy()
        return [list(np.argsort(-p)) for p in probs]

# --- 4) Runners for GA-only and Hybrid GAN–GA ---
def run_ga_only(pt, generations):
    ga = GeneticAlgorithm(pt)
    history = []
    for gen in range(generations):
        fits = ga.evaluate()
        best = min(fits)
        print(f"[GA-only] Gen {gen:2d} best = {int(best)}")
        history.append(best)
        ga.next_gen()
    return history

def run_hybrid(pt, generations, inject_every=5):
    ga = GeneticAlgorithm(pt)
    history = []
    for gen in range(generations):
        fits = ga.evaluate()
        best = min(fits)
        print(f"[Hybrid]  Gen {gen:2d} best = {int(best)}")
        history.append(best)
        new_samps = None
        if gen > 0 and gen % inject_every == 0:
            top_n = max(1, int(0.1 * ga.pop_size))
            top_idxs = np.argsort(fits)[:top_n]
            real_seqs = np.array([ga.population[i] for i in top_idxs], dtype=np.float32)
            gan = GAN(seq_len=pt.shape[0])
            gan.train(real_seqs)
            new_samps = gan.generate(5)
        ga.next_gen(new_samps)
    return history

# --- 5) Main & Comparison ---
if __name__ == '__main__':
    # Usage: python ai.py <instance_index> <generations>
    idx, generations = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 50)

    all_instances = load_orlib_flowshop('flowshop1.txt')
    print(f"Loaded {len(all_instances)} instances. Running instance #{idx} for {generations} generations.\n")

    pt = all_instances[idx]
    history_ga     = run_ga_only(pt, generations)
    history_hybrid = run_hybrid(pt, generations, inject_every=5)

    # Print table casting floats to ints
    print("\n| Generation | GA_only | Hybrid |")
    print("|------------|---------|--------|")
    for gen, ga_val, hyb_val in zip(range(generations), history_ga, history_hybrid):
        print(f"| {gen:10d} | {int(ga_val):7d} | {int(hyb_val):6d} |")

    # Plot convergence curves
    plt.figure()
    plt.plot(history_ga,    label='GA-only')
    plt.plot(history_hybrid, label='Hybrid GAN–GA')
    plt.xlabel('Generation')
    plt.ylabel('Best Makespan')
    plt.title('GA-only vs Hybrid GAN–GA Convergence')
    plt.legend()
    plt.show()

    # Plot final makespan comparison
    final_vals = {'GA-only': int(history_ga[-1]), 'Hybrid': int(history_hybrid[-1])}
    plt.figure()
    plt.bar(final_vals.keys(), final_vals.values())
    plt.ylabel('Final Best Makespan')
    plt.title('Final Makespan Comparison')
    plt.show()
