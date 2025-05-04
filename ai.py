import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import matplotlib.pyplot as plt

# ----------------------
# Step 1: Load PFSP instance (supports header with exactly two numbers or pure matrix)
def load_pfsp_instance(file_path):
    """
    Reads a PFSP instance. If the file's first line has exactly two integers, uses them as header;
    otherwise treats the file as a pure matrix of processing times.
    """
    with open(file_path, 'r') as f:
        first_line = f.readline().strip().split()
        # Treat as header only if exactly two numeric tokens
        if len(first_line) == 2 and all(token.isdigit() for token in first_line):
            n_jobs, n_machines = map(int, first_line)
            pt = []
            for _ in range(n_jobs):
                row = f.readline().strip().split()
                pt.append(list(map(int, row)))
            return np.array(pt, dtype=int)
        else:
            # reset pointer and load entire file as numeric matrix
            f.seek(0)
            return np.loadtxt(f, dtype=int)

# ----------------------
# Step 2: Genetic Algorithm
class GeneticAlgorithm:
    def __init__(self, processing_times, population_size=50, crossover_rate=0.8, mutation_rate=0.2):
        self.pt = processing_times
        self.num_jobs = processing_times.shape[0]
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = self._init_population()

    def _init_population(self):
        pop = []
        for _ in range(self.population_size):
            individual = list(range(self.num_jobs))
            random.shuffle(individual)
            pop.append(individual)
        return pop

    def fitness(self, individual):
        num_machines = self.pt.shape[1]
        comp = np.zeros((len(individual) + 1, num_machines + 1))
        for i, job in enumerate(individual, start=1):
            for m in range(1, num_machines + 1):
                comp[i, m] = max(comp[i-1, m], comp[i, m-1]) + self.pt[job, m-1]
        return comp[-1, -1]

    def evaluate_population(self):
        return [self.fitness(ind) for ind in self.population]

    def select(self, fitnesses):
        selected = []
        for _ in range(self.population_size):
            i, j = random.sample(range(self.population_size), 2)
            winner = self.population[i] if fitnesses[i] < fitnesses[j] else self.population[j]
            selected.append(winner.copy())
        return selected

    def crossover(self, p1, p2):
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))
        child = [-1]*size
        child[a:b] = p1[a:b]
        ptr = b
        for gene in p2[b:] + p2[:b]:
            if gene not in child:
                child[ptr % size] = gene
                ptr += 1
        return child

    def mutate(self, individual):
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]

    def create_new_generation(self, new_samples=None):
        fitnesses = self.evaluate_population()
        selected = self.select(fitnesses)
        next_pop = []
        # Inject GAN samples if provided (1-D sequences)
        if new_samples is not None:
            for seq in new_samples:
                if len(next_pop) < self.population_size:
                    next_pop.append(seq)
        # Fill remaining slots via GA
        while len(next_pop) < self.population_size:
            p1, p2 = random.sample(selected, 2)
            child = self.crossover(p1, p2) if random.random() < self.crossover_rate else p1.copy()
            if random.random() < self.mutation_rate:
                self.mutate(child)
            next_pop.append(child)
        self.population = next_pop

# ----------------------
# Step 3: GAN Definitions w/ custom training loops
def build_generator(latent_dim, seq_len):
    noise = Input(shape=(latent_dim,))
    x = layers.Dense(128, activation='relu')(noise)
    x = layers.Dense(256, activation='relu')(x)
    out = layers.Dense(seq_len, activation='softmax')(x)
    return Model(noise, out, name='generator')


def build_discriminator(seq_len):
    seq_in = Input(shape=(seq_len,))
    x = layers.Dense(256, activation='relu')(seq_in)
    x = layers.Dense(128, activation='relu')(x)
    validity = layers.Dense(1, activation='sigmoid')(x)
    return Model(seq_in, validity, name='discriminator')

class GAN:
    def __init__(self, seq_len, latent_dim=100):
        # Build models
        self.latent_dim = latent_dim
        self.gen = build_generator(latent_dim, seq_len)
        self.disc = build_discriminator(seq_len)
        # Define optimizers & loss
        self.opt_gen = tf.keras.optimizers.Adam()
        self.opt_disc = tf.keras.optimizers.Adam()
        self.bce = tf.keras.losses.BinaryCrossentropy()

    def train(self, real_seqs, epochs=500, batch_size=32):
        valid = tf.ones((batch_size, 1))
        fake = tf.zeros((batch_size, 1))
        for epoch in range(epochs):
            # Sample real
            idx = np.random.randint(0, real_seqs.shape[0], batch_size)
            real = tf.convert_to_tensor(real_seqs[idx], dtype=tf.float32)
            # Generate fake
            noise = tf.random.normal((batch_size, self.latent_dim))
            gen_seqs = self.gen(noise, training=False)
            
            # Train Discriminator
            with tf.GradientTape() as tape:
                pred_real = self.disc(real, training=True)
                loss_real = self.bce(valid, pred_real)
                pred_fake = self.disc(gen_seqs, training=True)
                loss_fake = self.bce(fake, pred_fake)
                loss_disc = 0.5 * (loss_real + loss_fake)
            grads_disc = tape.gradient(loss_disc, self.disc.trainable_weights)
            self.opt_disc.apply_gradients(zip(grads_disc, self.disc.trainable_weights))

            # Train Generator
            with tf.GradientTape() as tape:
                gen_seqs = self.gen(noise, training=True)
                validity = self.disc(gen_seqs, training=False)
                loss_gen = self.bce(valid, validity)
            grads_gen = tape.gradient(loss_gen, self.gen.trainable_weights)
            self.opt_gen.apply_gradients(zip(grads_gen, self.gen.trainable_weights))

            if epoch % 100 == 0:
                print(f"GAN Epoch {epoch}: D_loss={loss_disc:.4f}, G_loss={loss_gen:.4f}")

    def generate(self, n_samples):
        noise = tf.random.normal((n_samples, self.latent_dim))
        probs = self.gen(noise, training=False).numpy()
        return [list(np.argsort(-p)) for p in probs]

# ----------------------
# Step 4: Hybrid Loop

def hybrid_run(file_path, generations=50, gan_epochs=300, inject_interval=5):
    pt = load_pfsp_instance(file_path)
    ga = GeneticAlgorithm(pt)
    history = []

    for gen in range(generations):
        fitnesses = ga.evaluate_population()
        best = min(fitnesses)
        history.append(best)
        print(f"GA Generation {gen}, Best makespan: {best}")

        new_samples = None
        if gen > 0 and gen % inject_interval == 0:
            num_real = max(1, int(0.1 * ga.population_size))
            top_idxs = np.argsort(fitnesses)[:num_real]
            real_seqs = np.array([ga.population[i] for i in top_idxs], dtype=np.float32)
            gan = GAN(seq_len=ga.num_jobs)
            gan.train(real_seqs, epochs=gan_epochs)
            new_samples = gan.generate(n_samples=5)

        ga.create_new_generation(new_samples)

    plt.plot(history)
    plt.xlabel('Generation')
    plt.ylabel('Best Makespan')
    plt.title('Hybrid GAN–GA Convergence')
    plt.show()

if __name__ == '__main__':
    # Uncomment to generate a random test instance:
    # np.savetxt('random_pfsp_20x5.txt', np.random.randint(10,200,(20,5)), fmt='%d')
    hybrid_run('random_pfsp_20x5.txt')
