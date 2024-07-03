import numpy as np
import os
import pickle as pkl
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


def create_directory(directory_path):
    """
    Muller-Cleve, Simon F.; Istituto Italiano di Tecnologia - IIT; Event-driven perception in robotics - EDPR; Genova, Italy.
    """
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile! :(
            return None
        return directory_path


def train_validation_test_split(
    data,
    label,
    split=[70, 20, 10],
    seed=None,
    multiple=False,
    save_dataset=False,
    save_tensor=False,
    labels_type=None,
    labels_mapping=None,
    save_name=None,
    save_path=None
    ):
    """
    Creates train-validation-test splits using the sklearn train_test_split() twice.
    Can be used either to prepare "ready-to-use" splits or to create and store splits.

    If multiple splits are not needed and no saving option is set, the lists x_train, y_train, x_val, y_val, x_test, y_test are returned (without labels mapping).

    Function accepts lists, arrays, and tensor.
    Default split: [training: 70, validation: 20, test: 10]

    Fra, Vittorio; Politecnico di Torino; EDA Group; Torino, Italy.
    Muller-Cleve, Simon F.; Istituto Italiano di Tecnologia - IIT; Event-driven perception in robotics - EDPR; Genova, Italy.
    """

    if multiple:
        if (not save_dataset) & (not save_tensor):
            raise ValueError("Multiple train-val splits are created but no saving option is enabled.")

    if save_dataset | save_tensor:
        if (save_path == None) | (save_name == None):
            raise ValueError("Check a file name and a path are provided to save the datasets.")
        filename_prefix = save_path + save_name
        create_directory(save_path)

    # do some sanity checks first
    if len(split) != 3:
        raise ValueError(
            f"Split dimensions are wrong. Expected 3 but got {len(split)}. Please provide split in the form [train size, test size, validation size].")
    if min(split) == 0.0:
        raise ValueError(
            "Found entry 0.0. If you want to use only perfrom a two-folded split, use the sklearn train_test_split function only please.")
    if sum(split) > 99.0:
        split = [x/100 for x in split]
    if sum(split) < 0.99:
        raise ValueError("Please use a split summing up to 1, or 100%.")

    train, val, test = split
    split_1 = test
    split_2 = 1 - train/(train+val)

    x_trainval, x_test, y_trainval, y_test = train_test_split(
        data, label, test_size=split_1, shuffle=True, stratify=label, random_state=seed)


    if save_dataset: # Save the test split
        filename_test = filename_prefix + "_test"
        # xs test
        with open(f"{filename_test}.pkl", 'wb') as handle:
            pkl.dump(np.array(x_test, dtype=object), handle, protocol=pkl.HIGHEST_PROTOCOL)
        # ys test
        with open(f"{filename_test}_label.pkl", 'wb') as handle:
            pkl.dump(np.array(y_test, dtype=object), handle, protocol=pkl.HIGHEST_PROTOCOL)

    if save_tensor: # Save the test split
        filename_test = filename_prefix + "_ds_test"
        x_test = torch.as_tensor(np.array(x_test), dtype=torch.float)
        if labels_type == str:
            labels_test = torch.as_tensor(value2index(
                y_test, labels_mapping), dtype=torch.long)
        else:
            labels_test = torch.as_tensor(y_test, dtype=torch.long)
        ds_test = TensorDataset(x_test, labels_test)
        torch.save(ds_test, "{}.pt".format(filename_test))

    if multiple:
        for ii in range(10):
            x_train, x_val, y_train, y_val = train_test_split(
                x_trainval, y_trainval, test_size=split_2, shuffle=True, stratify=y_trainval, random_state=seed)

            if save_dataset:
                filename_train = filename_prefix + "_train"
                filename_val = filename_prefix + "_val"
                # xs training
                with open(f"{filename_train}_{ii}.pkl", 'wb') as handle:
                    pkl.dump(np.array(x_train, dtype=object), handle,
                                protocol=pkl.HIGHEST_PROTOCOL)
                # ys training
                with open(f"{filename_train}_{ii}_label.pkl", 'wb') as handle:
                    pkl.dump(np.array(y_train, dtype=object), handle,
                                protocol=pkl.HIGHEST_PROTOCOL)

                # xs validation
                with open(f"{filename_val}_{ii}.pkl", 'wb') as handle:
                    pkl.dump(np.array(x_val, dtype=object), handle,
                                protocol=pkl.HIGHEST_PROTOCOL)
                # ys validation
                with open(f"{filename_val}_{ii}_label.pkl", 'wb') as handle:
                    pkl.dump(np.array(y_val, dtype=object), handle,
                                protocol=pkl.HIGHEST_PROTOCOL)

            if save_tensor:
                filename_train = filename_prefix + "_ds_train"
                filename_val = filename_prefix + "_ds_val"

                x_train = torch.as_tensor(np.array(x_train), dtype=torch.float)
                if labels_type == str:
                    labels_train = torch.as_tensor(value2index(
                        y_train, labels_mapping), dtype=torch.long)
                else:
                    labels_train = torch.as_tensor(y_train, dtype=torch.long)

                x_validation = torch.as_tensor(
                    np.array(x_val), dtype=torch.float)
                if labels_type == str:
                    labels_validation = torch.as_tensor(value2index(
                        y_val, labels_mapping), dtype=torch.long)
                else:
                    labels_validation = torch.as_tensor(y_val, dtype=torch.long)

                ds_train = TensorDataset(x_train, labels_train)
                ds_val = TensorDataset(x_validation, labels_validation)

                torch.save(ds_train, "{}_{}.pt".format(filename_train,ii))
                torch.save(ds_val, "{}_{}.pt".format(filename_val,ii))

    else:
        x_train, x_val, y_train, y_val = train_test_split(
            x_trainval, y_trainval, test_size=split_2, shuffle=True, stratify=y_trainval, random_state=seed)
        
        if save_dataset:
            filename_train = filename_prefix + "_train"
            filename_val = filename_prefix + "_val"

            # xs training
            with open(f"{filename_train}.pkl", 'wb') as handle:
                pkl.dump(np.array(x_train, dtype=object), handle,
                            protocol=pkl.HIGHEST_PROTOCOL)
            # ys training
            with open(f"{filename_train}_label.pkl", 'wb') as handle:
                pkl.dump(np.array(y_train, dtype=object), handle,
                            protocol=pkl.HIGHEST_PROTOCOL)

            # xs validation
            with open(f"{filename_val}.pkl", 'wb') as handle:
                pkl.dump(np.array(x_val, dtype=object), handle,
                            protocol=pkl.HIGHEST_PROTOCOL)
            # ys validation
            with open(f"{filename_val}_label.pkl", 'wb') as handle:
                pkl.dump(np.array(y_val, dtype=object), handle,
                            protocol=pkl.HIGHEST_PROTOCOL)

        if save_tensor:
            filename_train = filename_prefix + "_ds_train"
            filename_val = filename_prefix + "_ds_val"
            x_train = torch.as_tensor(np.array(x_train), dtype=torch.float)
            if labels_type == str:
                labels_train = torch.as_tensor(value2index(
                    y_train, labels_mapping), dtype=torch.long)
            else:
                labels_train = torch.as_tensor(y_train, dtype=torch.long)

            x_validation = torch.as_tensor(
                np.array(x_val), dtype=torch.float)
            if labels_type == str:
                labels_validation = torch.as_tensor(value2index(
                    y_val, labels_mapping), dtype=torch.long)
            else:
                labels_validation = torch.as_tensor(y_val, dtype=torch.long)

            ds_train = TensorDataset(x_train, labels_train)
            ds_val = TensorDataset(x_validation, labels_validation)

            torch.save(ds_train, filename_train)
            torch.save(ds_val, filename_val)

        return x_train, y_train, x_val, y_val, x_test, y_test


def value2key(entry, dictionary):
    """
    Fra, Vittorio; Politecnico di Torino; EDA Group; Torino, Italy.
    """
    if (type(entry) != list) & (type(entry) != np.ndarray):
        key = [list(dictionary.keys())[list(dictionary.values()).index(entry)]]
    else:
        key = [list(dictionary.keys())[list(dictionary.values()).index(e)] for e in entry]
    return key


def index2key(entry, dictionary):
    """
    Fra, Vittorio; Politecnico di Torino; EDA Group; Torino, Italy.
    """
    if (type(entry) != list) & (type(entry) != np.ndarray):
        key = [list(dictionary.keys())[entry]]
    else:
        key = [list(dictionary.keys())[e] for e in entry]

    return key


def value2index(entry, dictionary):
    """
    Fra, Vittorio; Politecnico di Torino; EDA Group; Torino, Italy.
    """
    if (type(entry) != list) & (type(entry) != np.ndarray):
        idx = [list(dictionary.values()).index(entry)]
    else:
        idx = [list(dictionary.values()).index(e) for e in entry]

    return idx


def index2value(entry, dictionary):
    """
    Fra, Vittorio; Politecnico di Torino; EDA Group; Torino, Italy.
    """
    if (type(entry) != list) & (type(entry) != np.ndarray):
        value = [list(dictionary.values())[entry]]
    else:
        value = [list(dictionary.values())[e] for e in entry]

    return value


def training_loop(dataset, batch_size, net, optimizer, loss_fn, device):
    """
    Fra, Vittorio; Politecnico di Torino; EDA Group; Torino, Italy.
    """

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    batch_loss = []
    batch_acc = []

    for data, labels in tqdm(train_loader):

      data = data.to(device)#.swapaxes(1, 0)
      labels = labels.to(device)

      net.train()
      spk_rec, _= net(data) # CHANGED

      # Training loss
      loss_val = loss_fn(spk_rec, labels)
      batch_loss.append(loss_val.detach().cpu().item())

      # Training accuracy
      act_total_out = torch.sum(spk_rec, 0)  # sum over time
      _, neuron_max_act_total_out = torch.max(act_total_out, 1)  # argmax over output units to compare to labels
      batch_acc.append(np.mean((neuron_max_act_total_out == labels).detach().cpu().numpy()))

      # Gradient calculation + weight update
      optimizer.zero_grad()
      loss_val.backward()
      optimizer.step()

    epoch_loss = np.mean(batch_loss)
    epoch_acc = np.mean(batch_acc)

    return [epoch_loss, epoch_acc]


def val_test_loop(dataset, batch_size, net, loss_fn, device, shuffle=True, label_probabilities=False, return_spikes=False):
    """
    Fra, Vittorio; Politecnico di Torino; EDA Group; Torino, Italy.
    """

    with torch.no_grad():
      net.eval()

      loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

      batch_loss = []
      batch_acc = []

      for data, labels in tqdm(loader):
          data = data.to(device)#.swapaxes(1, 0)
          labels = labels.to(device)

          spk_out, _ = net(data)

          # Loss
          loss_val = loss_fn(spk_out, labels)
          batch_loss.append(loss_val.detach().cpu().item())

          # Accuracy
          act_total_out = torch.sum(spk_out, 0)  # sum over time
          _, neuron_max_act_total_out = torch.max(act_total_out, 1)  # argmax over output units to compare to labels
          batch_acc.append(np.mean((neuron_max_act_total_out == labels).detach().cpu().numpy()))

      if label_probabilities:
          log_softmax_fn = torch.nn.LogSoftmax(dim=-1)
          log_p_y = log_softmax_fn(act_total_out)
          if return_spikes:
            return [np.mean(batch_loss), np.mean(batch_acc)], torch.exp(log_p_y), spk_out.detach().cpu().numpy()
          else:
            return [np.mean(batch_loss), np.mean(batch_acc)], torch.exp(log_p_y)
      else:
        if return_spikes:
          return [np.mean(batch_loss), np.mean(batch_acc)], spk_out.detach().cpu().numpy()
        else:
          return [np.mean(batch_loss), np.mean(batch_acc)]