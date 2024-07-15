import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["JAX_PLATFORM_NAME"] = "GPU"
import jax
from jax import numpy as jnp
from jax import random, jit, lax
from jax.scipy.fft import dct, idct
from transformers import AutoTokenizer, OPTForCausalLM
from datasets import load_dataset
import pickle
import torch
device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from matplotlib import pyplot as plt
import itertools

import time
import logging  # Import logging module

###============ HYPERPARAMETERS ============

RNG = random.PRNGKey(0)

# Model names
# model_sizes = ["facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b", "facebook/opt-13b"]
# model_sizes = ["facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b", "facebook/opt-13b"]
# model_sizes = ["facebook/opt-125m"]
# model_sizes = ["facebook/opt-1.3b"]
model_sizes = ["facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b"]
kashin_max_iter = 200
kashin_tolerance = 1e-6
kmeans_max_iter = 300
kmeans_tolerance = 5e-6
kmeans_batch_size = 4096

Q_type = "Q" # "DCT", "Q"
inf_distance=False
kmeans_distance = "inf" if inf_distance else "2"

# Model sizes to evaluate
bits = ["k4", "k5", "k6"]  # Placeholder for quantization bits

# File path for the results.pkl file
file_path = 'results_clean_Q.pkl'

# Configure logging
logging.basicConfig(filename=f'quantization_without_emb_and_bads_{Q_type}_{kmeans_distance}.log', level=logging.INFO, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

###============ ============ ============

# Function to load and prepare the model and tokenizer
def load_model(model_name, quantization_bits=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

    if quantization_bits == 32:
        model = OPTForCausalLM.from_pretrained(model_name, device_map="auto")
    
    if quantization_bits == 16:
        model = OPTForCausalLM.from_pretrained(model_name, device_map="auto",
                                                torch_dtype=torch.float16)

    if quantization_bits == 8:
        model = OPTForCausalLM.from_pretrained(model_name, device_map="auto",
                                                load_in_8bit=True)
    
    if quantization_bits == 4:
        model = OPTForCausalLM.from_pretrained(model_name, device_map="auto",
                                                load_in_4bit=True)
        
    # Apply custom quantization for 'k' series
    if str(quantization_bits).startswith('k'):
        model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        bits = int(quantization_bits[1:])
        # for name, param in itertools.islice(model.named_parameters(), 10):
        for name, param in model.named_parameters():
            if param.requires_grad and len(param.data.size()) > 1 and 'embed' not in name:  # Check for weight matrices
                logging.info(f"💩 Compressing {name} layer of shape {param.data.shape}")
                modified_weight = kashin_zip(param.data, bits, layer_identifier=name, model_name=model_name)  # Apply your kashin_zip function
                logging.debug("After kashin weight type", modified_weight.dtype)
                param.data = modified_weight  # Replace original weight matrix
            elif len(param.data.size()) == 1 or 'embed' in name:
                param.data = param.data.to(dtype=torch.float16)  # Convert to torch.float16
                logging.debug("Avoiding kashin weight type", param.data.dtype)

    model.eval()

    return model, tokenizer

# Decompose matrix X with unit norm to the factors X = U + V, where U and 
# Q1.T @ V @ Q2 have small infinity norm
def decomposition_matrix_dct(X, max_iter=kashin_max_iter, tol=kashin_tolerance,
                              key=RNG):
    m, n = X.shape
    X_init = X.copy()
    U, V = jnp.zeros((m, n)), jnp.zeros((m, n))

    def projection(X, Y):
        # projection of X on Y
        return ((X.ravel()).T@Y.ravel()/jnp.linalg.norm(Y)**2)*Y

    iter = 0
    while iter < max_iter:
        # Y = Q1.T @ X @ Q2
        Y = (idct((idct(X, norm='ortho')).T, norm='ortho')).T
        if (jnp.linalg.norm(X.ravel(), 1) > jnp.linalg.norm(Y.ravel(), 1)):
            proj = projection(X, jnp.sign(X))
            U += proj
            X -= proj
        else:
            proj = projection(X, dct(dct(jnp.sign(Y), norm='ortho').T, norm='ortho').T)
            V += proj
            X -= proj

        # Check if termination condition is met
        residual = jnp.linalg.norm(X_init - U - V)
        if residual < tol:
            break

        iter += 1

    return X, U, V, residual, iter

def decomposition_matrix(X, max_iter=500, tol=1e-6, key=RNG):
    m, n = X.shape
    X_init = X.copy()
    U, V_ = jnp.zeros((m, n)), jnp.zeros((m, n))
    keys = random.split(key, 2)
    Q1, _ = jnp.linalg.qr(random.normal(keys[0], (m, m)))
    Q2, _ = jnp.linalg.qr(random.normal(keys[1], (n, n)))

    def projection(X, Y):
        # projection of X on Y
        return ((X.ravel()).T@Y.ravel()/jnp.linalg.norm(Y)**2)*Y

    iter = 0
    while iter < max_iter:
        Y = Q1.T @ X @ Q2
        if (jnp.linalg.norm(X.ravel(), 1) > jnp.linalg.norm(Y.ravel(), 1)):
            proj = projection(X, jnp.sign(X))
            U += proj
            X -= proj
        else:
            proj = projection(X, Q1 @ jnp.sign(Y) @ Q2.T)
            V_ += proj
            X -= proj

        # Check if termination condition is met
        residual = jnp.linalg.norm(X_init - U - V_)
        if residual < tol:
            break

        iter += 1

    return X, U, V_, residual, iter, Q1, Q2

def plot_and_save_centroids(data, centroids, bit_label, layer_identifier, model_name=''):
    # z = gaussian_kde(data)(data)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5, label="Entries")
    plt.scatter(centroids[:, 0], centroids[:, 1], color='red', label='Centroids')
    plt.title(f'Density Scatter Plot of Stacked U and V.\nLayer {layer_identifier}')
    plt.xlabel('U entries')
    plt.ylabel('V entries')
    plt.legend(bbox_to_anchor=(0.5, 0), loc="lower right", bbox_transform=plt.gcf().transFigure, ncol=2)

    
    folder_name = f'do_not_compress_bad_layers_{model_name}'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    filename = f'{folder_name}/layer_{layer_identifier}_centroids_{bit_label}.png'
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def initialize_centroids(data, n_clusters, init):
    """Assume init is correctly shaped and directly return it."""
    return init

@jit
def euclidean_distances(X, Y):
    """Compute the squared Euclidean distances between each point in X and Y."""
    X_square = jnp.sum(X ** 2, axis=1, keepdims=True)
    Y_square = jnp.sum(Y ** 2, axis=1)
    XY = jnp.dot(X, Y.T)
    distances = X_square - 2 * XY + Y_square
    return distances

@jit
def infinity_norm_distances(X, Y):
    """Compute the infinity norm (max abs value) distances between each point in X and Y."""
    # Expand dimensions of X and Y for broadcasting to compute pairwise differences
    X_expanded = X[:, None, :]  # Shape becomes (X.shape[0], 1, X.shape[1])
    Y_expanded = Y[None, :, :]  # Shape becomes (1, Y.shape[0], Y.shape[1])
    
    # Compute the absolute differences
    abs_diff = jnp.abs(X_expanded - Y_expanded)
    
    # Find the maximum absolute difference across all dimensions for each pair
    distances = jnp.max(abs_diff, axis=2)
    
    return distances

@jit
def assign_clusters(data, centroids, inf_distance=inf_distance):
    """Assign data points to the nearest cluster based on specified distance."""
    def euclidean_true_fun(_):
        return euclidean_distances(data, centroids)
    
    def infinity_norm_true_fun(_):
        return infinity_norm_distances(data, centroids)
    
    # Use cond to conditionally choose the distance calculation
    distances = lax.cond(inf_distance,
                         infinity_norm_true_fun,
                         euclidean_true_fun,
                         None)
    
    return jnp.argmin(distances, axis=1)

def update_centroids(data, labels, n_clusters, centroids_old):
    """Update centroids based on current cluster assignment, handling empty clusters."""
    new_centroids = jnp.zeros_like(centroids_old)
    for i in range(n_clusters):
        points_in_cluster = data[jnp.where(labels == i)]
        if points_in_cluster.shape[0] > 0:
            new_centroids = new_centroids.at[i].set(jnp.mean(points_in_cluster, axis=0))
        else:
            # Keep the centroid unchanged for empty clusters
            new_centroids = new_centroids.at[i].set(centroids_old[i])
    return new_centroids

@jit
def centroids_distance(centroids_old, centroids_new):
    """Calculate the distance between the old and new centroids."""
    return jnp.sqrt(jnp.sum((centroids_old - centroids_new) ** 2))

def mini_batch_kmeans(data, n_clusters, init, max_iter=kmeans_max_iter, batch_size=kmeans_batch_size, tol=kmeans_tolerance):
    """Mini-batch KMeans clustering with tolerance and maximum iteration stopping criteria."""
    centroids = initialize_centroids(data, n_clusters, init)
    for iteration in range(max_iter):
        # Randomly sample a batch from the data
        batch_indices = jax.random.choice(jax.random.PRNGKey(iteration), data.shape[0], shape=(batch_size,), replace=False)
        batch_data = data[batch_indices]

        # Assign clusters and update centroids based on the batch
        labels = assign_clusters(batch_data, centroids, inf_distance=True)
        new_centroids = update_centroids(batch_data, labels, n_clusters, centroids)
        
        # Check for convergence
        d_clusters = centroids_distance(centroids, new_centroids)
        if d_clusters < tol:
            break  # Convergence criterion met
        
        centroids = new_centroids
    
    logging.info(f"k means finished wih {iteration} iters and tol {d_clusters:.2e}")
    # Final assignment on all data
    final_labels = assign_clusters(data, centroids)
    return centroids, final_labels


def kashin_zip(X_, bits, layer_identifier="", device=device_t, 
               kashin_max_iter=kashin_max_iter,
               tol=kashin_tolerance,
               model_name="",
               Q_type=Q_type):
    X = jnp.array(X_.cpu().numpy())
    start_time = time.time()  # Start timing
    X_norm = jnp.linalg.norm(X)
    X /= X_norm

    number_of_clusters = int(2**(bits))

    if Q_type == "Q":
        _, U, V_, residual, iter, Q1, Q2 = decomposition_matrix(X, max_iter=kashin_max_iter, tol=tol)
    elif Q_type == "DCT":
        _, U, V_, residual, iter = decomposition_matrix_dct(X, max_iter=kashin_max_iter, tol=tol)
    if iter == kashin_max_iter:
        if Q_type == "Q":
            logging.info(f"🤓 Q decomposition avoided with residual{residual:.2e} after {iter} iterations")
        elif Q_type == "DCT":
            logging.info(f"🤓 DCT decomposition avoided with residual{residual:.2e} after {iter} iterations")
        return X_
    
    if Q_type == "Q":
        logging.info(f"🤓 Q decomposition completed with  residual{residual:.2e} after {iter} iterations")
        QVQ = Q1.T @ V_ @ Q2
    elif Q_type == "DCT":
        logging.info(f"🤓 DCT decomposition completed with  residual{residual:.2e} after {iter} iterations")
        QVQ = (idct((idct(V_, norm='ortho')).T, norm='ortho')).T

    # Flatten and stack U and V_transformed
    U_flattened = U.ravel()
    QVQ_flattened = QVQ.ravel()
    U_QVQ_stacked = jnp.column_stack((U_flattened, QVQ_flattened))

    # Logging time after decomposition
    decomposition_time = time.time()
    logging.info(f"Decomposition completed in {(decomposition_time - start_time):.2f} seconds")

    # KMeans clustering
    # Uniform grid for initialization
    u_min, u_max = U_flattened.min(), U_flattened.max()
    v_min, v_max = QVQ_flattened.min(), QVQ_flattened.max()
    grid_x, grid_y = jnp.meshgrid(jnp.linspace(u_min, u_max, int(jnp.sqrt(number_of_clusters))),
                                 jnp.linspace(v_min, v_max, int(jnp.sqrt(number_of_clusters))))
    init_points = jnp.column_stack([grid_x.ravel(), grid_y.ravel()])


    centroids, labels = mini_batch_kmeans(
        data=U_QVQ_stacked, 
        n_clusters = number_of_clusters,
        init=init_points
        )

    # Logging time after clustering
    clustering_time = time.time()
    plot_and_save_centroids(U_QVQ_stacked, centroids, bits, layer_identifier, model_name=model_name)
    logging.info(f"2d Clustering completed in {(clustering_time - decomposition_time):.2f} seconds")


    # Replace all elements in the U matrix and QVQ matrix with the closest centroids from clusters_UV[:,:, 0] and clusters_UV[:,:, 1] correspondingly
    U_q = centroids[labels, 0].reshape(U.shape)
    QVQ_q = centroids[labels, 1].reshape(QVQ.shape)

    # Logging time after clustering
    quant_time = time.time()
    logging.info(f"Quantization completed in {(quant_time - clustering_time):.2f} seconds")

    # Unzip everything together
    if Q_type == "Q":
        X_q = (U_q + Q1 @ QVQ_q @ Q2.T) * X_norm
    elif Q_type == "DCT":
        X_q = (U_q + dct(dct(QVQ_q, norm='ortho').T, norm='ortho').T) * X_norm
    

    end_time = time.time()
    logging.info(f"Total time for kashin_zip: {(end_time - start_time):.2f} seconds")

    return torch.from_numpy(np.array(X_q)).to(device).to(torch.float16)
    
# Function to calculate perplexity
# Ensure your CUDA device is available
def calculate_perplexity(model, tokenizer, text):
    encodings = tokenizer(text, return_tensors='pt')
    max_length = model.config.max_position_embeddings  # Changed from n_positions to max_position_embeddings
    stride = 512

    nlls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i    # may be different from stride on last loop
        input_ids = encodings.input_ids[:,begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:,:-trg_len] = -100

        with torch.no_grad():
            logging.debug("PPL inputs and target", input_ids.dtype, target_ids.dtype)
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs.loss * trg_len

        nlls.append(log_likelihood.to(torch.float32))
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()

# Function to read the existing data from the file
def read_existing_data(file_path):
    try:
        with open(file_path, 'rb') as fp:
            existing_data = pickle.load(fp)
        return existing_data
    except FileNotFoundError:
        return {}

# Function to append new data to the existing data
def append_new_data(existing_data, new_data):
    for model_size in new_data:
        if model_size not in existing_data:
            existing_data[model_size] = {}
        for bit in new_data[model_size]:
            existing_data[model_size][bit] = new_data[model_size][bit]
    return existing_data

# Example usage
existing_data = read_existing_data(file_path)
# You would replace new_data with your actual new data
new_data = {}  # Replace this with your actual new data
updated_data = append_new_data(existing_data, new_data)

# Save the updated data back to the file
with open(file_path, 'wb') as fp:
    pickle.dump(updated_data, fp)

# Load LAMBADA dataset
lambada_dataset = load_dataset('lambada', split='test')
lambada_data = ' '.join(lambada_dataset['text'])

results = {}

for model_size in model_sizes:
    existing_data = read_existing_data(file_path)
    new_data = {}
    for bit in bits:
        new_data[model_size] = {}
        if bit.startswith("k"):
            logging.info(f"\n💎 Benchmarking {model_size} Kashin Quantized {bit}")
        else:
            logging.info(f"\n💎 Benchmarking {model_size} {'Quantized ' + str(bit) + ' bit' if bit<32 else ''}")
        # Load model - assuming proper quantization setup
        model, tokenizer = load_model(model_size, quantization_bits=bit)
        ppl = calculate_perplexity(model, tokenizer, lambada_data)
        new_data[model_size][bit] = ppl
        existing_data = append_new_data(existing_data, new_data)
        logging.info(f"💎 Perplexity {ppl}")
        # save dictionary to file
        with open('results.pkl', 'wb') as fp:
            pickle.dump(existing_data, fp)