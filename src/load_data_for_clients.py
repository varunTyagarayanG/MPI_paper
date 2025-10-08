import math
import random
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.util_functions import set_seed, create_data, numpy_to_tensor, load_data

def dist_data_per_client(data_path, dataset_name, num_clients, batch_size, non_iid_per, device):
    set_seed(27)
    print("\nPreparing Data")
    train_data, test_data = create_data(data_path, dataset_name)
    X_train = train_data.data
    Y_train = np.array(train_data.targets)
    
    # Convert to numpy
    X_train = np.array(X_train)

    print("\nDividing the data among clients")

    # Find all the class labels
    classes = list(np.unique(Y_train))
    classes.sort()

    step = math.ceil(100 / len(classes))
    min_client_in_chunk = 1  # safer default for CIFAR100

    client_data_feats = [list() for _ in range(num_clients)]
    client_data_labels = [list() for _ in range(num_clients)]

    inter_non_iid_score = int((non_iid_per * 100) / step)
    intra_non_iid_score = int((non_iid_per * 100) % step)

    class_chunks = []
    tmp = []
    for index, class_ in enumerate(classes):
        indices = np.arange(index, index + inter_non_iid_score) % len(classes)
        class_chunk = list(set(classes) - set(np.array(classes)[indices]))
        class_chunk.sort()
        class_chunks.append(class_chunk)
        tmp.extend(class_chunk)

    # ==============================
    # FIXED CLIENT DISTRIBUTION
    # ==============================
    clients_per_chunk = [1] * len(class_chunks)   # at least one client per chunk
    remaining_clients = num_clients - len(class_chunks)
    for _ in range(remaining_clients):
        clients_per_chunk[random.randint(0, len(class_chunks) - 1)] += 1

    cumulative_clients_per_chunk = [sum(clients_per_chunk[:i+1]) for i in range(len(clients_per_chunk))]

    class_count_dict = {class_: 0 for class_ in classes}

    for index, class_chunk in enumerate(class_chunks):
        for class_label in class_chunk:
            indices = np.where(Y_train == class_label)[0]
            start = round(class_count_dict[class_label] * (len(indices) / Counter(tmp)[class_label]))
            end = round((class_count_dict[class_label] + 1) * (len(indices) / Counter(tmp)[class_label]))
            class_count_dict[class_label] += 1
            indices = indices[start:end]

            num_data_per_client = math.ceil(len(indices) / clients_per_chunk[index])
            last_client_data = len(indices) % clients_per_chunk[index]

            val_last_client = 5
            x1, x2 = 1, clients_per_chunk[index]
            y1, y2 = num_data_per_client + last_client_data - val_last_client, val_last_client
            min_m, min_c = 0, val_last_client

            if x2 == x1:  # only one client in this chunk
                m, c = 0, y1   # slope=0, client gets all samples
            else:
                max_m = (y2 - y1) / (x2 - x1)
                max_c = y1 - (max_m * x1)
                m = min_m + (((max_m - min_m) / (x2 - x1)) * intra_non_iid_score)
                c = min_c + (((max_c - min_c) / (x2 - x1)) * intra_non_iid_score)

            agg_points = 0

            denom = sum([m * (i + 1) + c for i in range(clients_per_chunk[index])])
            # prevent division by zero if denom is zero
            if denom == 0:
                weights = [1.0 / clients_per_chunk[index]] * clients_per_chunk[index]
            else:
                weights = [(m * (i + 1) + c) / denom for i in range(clients_per_chunk[index])]

            client_index_start = cumulative_clients_per_chunk[index - 1] if index > 0 else 0
            client_index_end = cumulative_clients_per_chunk[index]

            for index_count, i in enumerate(np.arange(client_index_start, client_index_end)):
                if i >= num_clients:
                    break
                else:
                    num_points = weights[index_count] * len(indices)
                    data = X_train[indices[round(agg_points):round(agg_points + num_points)]]
                    labels = [class_label for _ in range(len(data))]
                    client_data_feats[i].extend(data)
                    client_data_labels[i].extend(labels)
                    agg_points += num_points

    client_loaders = []
    for i in range(num_clients):
        x = numpy_to_tensor(np.asarray(client_data_feats[i]), device, "float")
        y = numpy_to_tensor(np.asarray(client_data_labels[i]), device, "long")
        dataset = load_data(x, y)
        client_loaders.append(DataLoader(dataset=dataset, batch_size=x.shape[0], shuffle=True, num_workers=0))

    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=0)

    return client_loaders, test_loader
