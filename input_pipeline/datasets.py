import os
import gin
import logging
import gin.config
import tensorflow as tf
import glob
import ast
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from input_pipeline.preprocessing import augment


@gin.configurable
def load(name,
         data_dir,
         data_dir_train,
         data_dir_test,
         val_split,
         ts_name, 
         win_size, 
         win_offset):
    '''
    Load the dataset, splits it into training, validation and test set and 
    returns TensorFlow datasets. 

        Parameters:
                name (str): Name of the dataset
                data_dir (str): Path to directory containing the datasets
                data_dir_train (str): Path to train data inside NASA SMP&MSL dataset directory
                data_dir_test (str): Path to test data inside NASA SMP&MSL directory
                val_split (int): Ration to split train data into training and validation sets
                ts_name (str): Name of the telemetry channel used ("all", "A-1", ...)
                win_size (int): Length of the window
                win_offset (int): Shift of the sliding window                
        
        Returns:
                Tuple [tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Optional[dict]]:
                    A tuple containing:
                        - 'ds_train': Training dataset
                        - 'ds_val': Validation dataset
                        - 'ds_test': Testing dataset
                        - 'ds_info': Optional metadata of information about the dataset (different for NASA dataset)
    '''
    if name == "nasa":
        logging.info(f"Preparing dataset {name}...")
        
        if ts_name == "all":
            # Split train into train and validation
            pa_train, pa_val = train_test_split(glob.glob(os.path.join(data_dir, data_dir_train,"*.npy")), test_size=val_split, random_state=42)
            pa_test = glob.glob(os.path.join(data_dir, data_dir_test,"*.npy"))

            # Create tf.data.Dataset 
            ds_info = {}
            ds_train, ds_info['train'] = create_dataset(pa_train, win_size, win_offset)
            ds_val, ds_info['val'] = create_dataset(pa_val, win_size, win_offset)
            ds_test, ds_info['test'] = create_dataset(pa_test, win_size, win_size)

            return prepare(ds_train, ds_val, ds_test, ds_info) 
            
        else:
            pa_train = os.path.join(data_dir, data_dir_train, ts_name + ".npy")
            pa_test = os.path.join(data_dir, data_dir_test, ts_name + ".npy")

            # Create tf.data.Dataset for training and validation
            data_train = np.load(pa_train)[:,0]
            start_val = int((1-val_split) * len(data_train))
            ds_train = tf.data.Dataset.from_tensor_slices(data_train[:start_val]).window(win_size, shift=win_offset, drop_remainder=True).flat_map(lambda window: window.batch(win_size)) 
            ds_val = tf.data.Dataset.from_tensor_slices(data_train[start_val:]).window(win_size, shift=win_offset, drop_remainder=True).flat_map(lambda window: window.batch(win_size)) 

            # Create tf.data.Dataset for evaluation
            data_test = np.load(pa_test)[:,0]
            ds_test = tf.data.Dataset.from_tensor_slices(data_test).window(win_size, shift=win_size, drop_remainder=True).flat_map(lambda window: window.batch(win_size)) 

            ds_info = {}
            l_train = len(data_train[:start_val]) # length before windowing
            l_val = len(data_train[start_val:])
            l_test = len(data_test)

            # Get true anomaly position and type
            labels = pd.read_csv(os.path.join(data_dir, 'NASA', 'labeled_anomalies.csv'), converters={"anomaly_sequences": ast.literal_eval})
            class_label = labels.loc[labels["chan_id"]==ts_name, "class"].iloc[0].replace("[", "['").replace("]", "']").replace(", ", "', '") # convert "[contextual, contextual]" to "['contextual', 'contextual']"

            ds_info['train'] = {"length": l_train, 
                                "win_size": win_size, 
                                "win_offset": win_offset} 
            ds_info['val'] = {"length": l_val, 
                                "win_size": win_size, 
                                "win_offset": win_offset} 
            ds_info['test'] = {"length": l_test, 
                                "win_size": win_size, 
                                "win_offset": win_offset,
                                "seq": labels.loc[labels["chan_id"]==ts_name, "anomaly_sequences"].iloc[0], # anomaly position
                                "class": ast.literal_eval(class_label)} # anomaly type

            return prepare(ds_train, ds_val, ds_test, ds_info)
    else:
        raise ValueError


@gin.configurable
def create_dataset(file_paths, win_size, win_offset):
    '''
    Create a Tenserflow dataset from a list of file paths and apply a sliding window approach.

        Parameters:
                file_paths (list): List of paths to the npy files used for the dataset
                win_size (int): Length of the window
                win_offset (int): Shift of the sliding window

        Returns:
                dataset (tf.data.Dataset): A Tenserflow dataset 
                info (list): A list of tuples -> (win_count, lost_samples, l) for each time series in dataset
    '''
    # Create tf.data.Dataset
    info = [] # list of tupples
    data = []
    for file in file_paths:
        d = np.load(file)[:,0] # only telemetry channel is used
        l = len(d)
        win_count = ((l - win_size) // win_offset) + 1 # number of windows in one signal
        lost_samples = l % win_offset # number of samples lost
        info.append((win_count, lost_samples, l))
        data.append(d) 

    dataset = tf.data.Dataset.from_generator(lambda: iter(data), output_signature=tf.TensorSpec(shape=(None,), dtype=tf.float32))
    dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x).window(win_size, shift=win_offset, drop_remainder=True)) # creating windows (windows are datasets)
    dataset = dataset.flat_map(lambda window: window.batch(win_size)) # flat_map and batch resolve nested dataset 

    return dataset, info


@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching, repeat, augmentation):
    '''
    Prepares Tensorflow datasets for training, validation and testing by applying
    batching, caching, prefetching and augmentation.

        Parameters:
                ds_train (tf.data.Dataset): Training dataset
                ds_val (tf.data.Dataset): Validation dataset
                ds_test (tf.data.Dataset): Test dataset
                ds_info (tfds.core.DatasetInfo): Optional metadata of information about the dataset
                batch_size (int): Size of each batch
                caching (bool): Whether to cache the datasets for improved performance
                repeat (bool): Whether to repeat the dataset 
                augmentation (bool): Whether to augment the dataset 
        
        Returns:
                ds_train (tf.data.Dataset): Prepared training dataset
                ds_val (tf.data.Dataset): Prepared validation dataset
                ds_test (tf.data.Dataset): Prepared test dataset
                ds_info(tfds.core.DatasetInfo): Optional metadata of information about the dataset
    '''
    # Prepare training dataset
    if caching:
        ds_train = ds_train.cache()
    if augmentation:
        ds_train = ds_train.map(
            augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.batch(batch_size)
    if repeat:
        ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info