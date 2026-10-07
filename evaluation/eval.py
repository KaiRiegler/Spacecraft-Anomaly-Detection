import gin
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import logging
import wandb
from evaluation.metrics import Metrics
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, balanced_accuracy_score

def evaluate(model, checkpoint, ds_test, ds_info, run_paths):
    '''
    Evaluate the given model on the test dataset and log metrics

        Parameter:
                model (tf.keras.Model): The trained model to evaluate
                checkpoint (tf.train.Checkpoint): The checkpoint manager for restoring model weights
                ds_test (tf.data.Dataset): The test dataset
                ds_info (dict): Information about the dataset
                run_paths (dict): Dictionary containing paths for checkpoints and logging

        Returns:
                None
    '''
    manager = tf.train.CheckpointManager(checkpoint, run_paths['path_ckpts_train'], max_to_keep=1)
    checkpoint.restore(manager.latest_checkpoint).expect_partial()
    logging.info(f'Evaluation on {model.name} with {manager.latest_checkpoint}.')

    eval_metrics = Metrics("eval")
    eval_metrics.reset_states()

    reconstruction_errors = []
    reconstruction_values = []
    input_values = []

    for idx, inputs in enumerate(ds_test):
        reconstruction_loss, reconstruction = eval_step(inputs, model)

        reconstruction_errors.append(reconstruction_loss)
        reconstruction_values.append(reconstruction)
        input_values.append(inputs)

        eval_metrics.update_state(reconstruction_loss)

    # Flatten and concatenate all batches of the reconstruction error
    flattened_reconstruction_errors = flatten_and_convert_to_np_array(reconstruction_errors)
    flattened_reconstruction = flatten_and_convert_to_np_array(reconstruction_values)
    flattened_inputs = flatten_and_convert_to_np_array(input_values)

    anomalies, threshold = detect_anomalies(flattened_reconstruction_errors, len(flattened_inputs))
    compute_anomaly_metrics(anomalies, ds_info)

    plot_reconstruction_errors_with_threshold(flattened_reconstruction_errors,
                                              flattened_reconstruction,
                                              flattened_inputs,
                                              anomalies,
                                              threshold,
                                              ds_info)

    template = 'Evaluation: Reconstruction Error: {}'
    logging.info(template.format(eval_metrics.result("loss")))
    
    # Write summary to wandb
    wandb.log({'mean reconstruction error': eval_metrics.result("loss")})

    return None


@tf.function
def eval_step(inputs, model):
    '''
    Perform a single evaluation step

        Parameters:
                inputs (Tensor): A batch of input data
                model (tf.keras.Model): The trained model

        Returns:
                reconstruction loss (Tensor): Reconstruction loss of one batch for one eval step
                reconstruction (Tensor): Predicted vaules of one batch for one eval step
    '''
    reconstruction = model(inputs, training=False)
    inputs = tf.expand_dims(inputs, axis=-1)

    reconstruction_loss = tf.keras.losses.mean_squared_error(inputs, reconstruction) 

    return reconstruction_loss, reconstruction


@gin.configurable
def detect_anomalies(reconstruction_errors, signal_length, threshold):
    '''
    Detect anomalies in the reconstruction errors using a threshold

        Parameters:
                reconstruction_errors (np.ndarray): Array of reconstruction errors
                signal_length (int): Length of the data signal
                threshold (float): Threshold for detecting anomylies

        Returns:
                anomalies (np.array): Indices of detected anomalies
                threshold (float): Threshold for anomaly detection
    '''
    anomalies = np.where(reconstruction_errors > threshold)[0]

    if len(anomalies) > 0:
        logging.info(f'Anomalies detected')
    else:
        logging.info(f'No Anomalies detected')

    return anomalies, threshold


def compute_anomaly_metrics(anomalies, ds_info):
    '''
    Compute different metrics to evaluate the detection. 

        Parameters:
                anomalies (np.ndarray): Indices of detected anomalies
                ds_info (dict): Information of dataset. (defined in dataset.py)

        Returns:
                None
    '''    
    y_true = np.zeros(ds_info["test"]["length"])
    for a in ds_info["test"]["seq"]:
        y_true[a[0]:a[1]] = 1

    y_pred = np.zeros(ds_info["test"]["length"])
    y_pred[anomalies] = 1

    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    b_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    template = 'Detection Metrics: \nTN:{0}, FP:{1}\nFN:{2}, TP:{3}\nAccuracy: {4}, Balanced Accuracy: {5}, F1: {6}'
    logging.info(template.format(cm[0][0], cm[0][1], cm[1][0], cm[1][1], acc, b_acc, f1))

    return None


def plot_reconstruction_errors_with_threshold(flattened_reconstruction_errors,
                                              flattened_reconstruction,
                                              flattened_inputs,
                                              anomalies,
                                              threshold,
                                              ds_info):
    '''
    Plot reconstruction errors and compare true vs reconstructed signals

        Parameters:
                flattened_reconstruction_errors (np.ndarray): Array of reconstruction errors
                flattened_reconstruction (np.ndarray): Array of reconstructed signal values
                flattened_inputs (np.ndarray): Array of true signal values
                anomalies (np.ndarray): Indices of detected anomalies
                threshold (float): Anomaly detection threshold
                ds_info (dict): Information about the dataset

        Returns:
                None
    '''
    # Figure 1 Histogram
    plt.figure(1, figsize=(15, 6))
    plt.hist(np.array(flattened_reconstruction_errors), bins=100)
    plt.title('Reconstruction Error Distribution')
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Amount of Samples")
    plt.grid(True)
    wandb.log({"Histogram": wandb.Image(plt)})
    plt.close()

    # Figure 2 Reconstruction
    plt.figure(2, figsize=(15, 6))
    indices = np.arange(1, len(flattened_reconstruction_errors) + 1)
    plt.plot(indices, np.array(flattened_reconstruction_errors), marker='o', linestyle='-', color='blue', label="Reconstruction Error")
    plt.axhline(y=threshold, color='red', linestyle='--', label=f"Threshold = {threshold:.2f}")
    plt.scatter(indices[anomalies], np.array(flattened_reconstruction_errors)[anomalies], color='orange', label="Anomalies", zorder=5)
    # Highlight true anomaly area
    for a in ds_info["test"]["seq"]:
        plt.axvspan(a[0], a[1], color='green', alpha=0.3, label="True Anomaly Area" if a == ds_info["test"]["seq"] else "")
    plt.title("Reconstruction Errors with Anomalies")
    plt.xlabel("Sample Index")
    plt.ylabel("Reconstruction Error")
    plt.legend()
    plt.grid(True)
    wandb.log({"Reconstruction": wandb.Image(plt)})
    plt.close()

    # Figure 3 Plot True and Reconstructed Signal
    plt.figure(3, figsize=(15, 6))
    plt.plot(np.array(flattened_inputs), label="True Signal", color='green')
    plt.plot(np.array(flattened_reconstruction), label="Reconstructed Signal", color='blue')
    # Highlight true anomaly area
    for a in ds_info["test"]["seq"]:
        plt.axvspan(a[0], a[1], color='green', alpha=0.3, label="True Anomaly Area" if a == ds_info["test"]["seq"] else "")
    # Highlight anomaly areas
    for anomaly in anomalies:
        plt.axvspan(anomaly - 0.3, anomaly + 0.3, color='red', alpha=0.2, label="Anomaly Area" if anomaly == anomalies[0] else "")
    plt.title("Test: True vs. Reconstruction")
    plt.xlabel("Sample Index")
    plt.ylabel("Signal Value")
    plt.legend()
    plt.grid(True)
    wandb.log({"True and Reconstructed Signal": wandb.Image(plt)})
    plt.close()


def flatten_and_convert_to_np_array(tf_tensor):
    '''
    Flatten a tensor or array and convert it to a NumPy array.

        Parameters:
                tf_tensor (list or tf.Tensor): Tensor or list of arrays to flatten.

        Returns:
                flattened_array (np.ndarray): Flattened NumPy array.
    '''
    flattened_array = np.concatenate([
        batch.numpy().reshape(-1) if hasattr(batch, "numpy") else np.array(batch).reshape(-1)
        for batch in tf_tensor
        ])
    
    return flattened_array
