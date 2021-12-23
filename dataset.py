from torch.utils.data import Dataset, DataLoader, Sampler
import torch
import random


class NumpyDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


class BertDataset(Dataset):
    def __init__(self, x, y, tokenizer, length=128):
        super(BertDataset, self).__init__()
        self.tokenizer = tokenizer
        self.length = length
        self.x = x
        self.y = torch.tensor(y)
        self.tokens_cache = {}

    def tokenize(self, x):
        dic = self.tokenizer.batch_encode_plus(
                    [x],        # input must be a list
                    max_length=self.length,
                    padding='max_length',
                    truncation=True,
                    return_token_type_ids=True,
                    return_tensors="pt"
                )
        return [x[0] for x in dic.values()]     # get rid of the first dim

    def __getitem__(self, idx):
        if idx not in self.tokens_cache:
            self.tokens_cache[idx] = self.tokenize(self.x[idx])
        input_ids, token_type_ids, attention_mask = self.tokens_cache[idx]
        return input_ids, token_type_ids, attention_mask, self.y[idx]

    def __len__(self):
        return len(self.y)


class TrainSampler(Sampler):
    def __init__(self, dataset, batch_size, sim_ratio=0.5):
        super().__init__(None)
        self.dataset = dataset
        self.batch_size = batch_size
        self.x = dataset.x
        self.y = dataset.y
        self.sim_ratio = sim_ratio
        self.num_pos_samples = int(batch_size * sim_ratio)
        print(f'train sampler with batch size = {batch_size} and postive sample ratio = {sim_ratio}')

        self.length = len(list(self.__iter__()))

    def __iter__(self):
        indices = list(range(len(self.y)))
        label_cluster = {}
        for i in indices:
            label = self.y[i].item()
            if label not in label_cluster:
                label_cluster[label] = []
            label_cluster[label].append(i)
        for key, value in label_cluster.items():
            random.shuffle(value)

        assert len(label_cluster[0]) > self.num_pos_samples, \
            f"only {len(label_cluster[0])} samples in each class, but {self.num_pos_samples} pos samples needed"

        # too time-consuming, i.e., O(|D||C|/|B|)s
        batch_indices = []
        flag = True
        while flag:
            # find a valid positive sample class
            available_classes = list(filter(lambda x: len(label_cluster[x]) >= self.num_pos_samples,
                                            list(range(max(self.y) + 1))))
            if len(available_classes) == 0:
                break
            class_count = random.choice(available_classes)

            # fill in positive samples
            batch_indices.append(label_cluster[class_count][-self.num_pos_samples:])
            del label_cluster[class_count][-self.num_pos_samples:]

            # fill in negative samples
            for i in range(self.batch_size - self.num_pos_samples):
                available_classes = list(filter(lambda x: len(label_cluster[x]) > 0, list(range(max(self.y) + 1))))
                if class_count in available_classes:
                    available_classes.remove(class_count)
                if len(available_classes) == 0:
                    flag = False
                    break
                rand_class = random.choice(available_classes)
                batch_indices[-1].append(label_cluster[rand_class].pop())

            random.shuffle(batch_indices[-1])

        random.shuffle(batch_indices)
        all = sum(batch_indices, [])

        return iter(all)

    def __len__(self):
        return self.length


class TrainSamplerMultiClass(Sampler):
    def __init__(self, dataset, batch_size, num_classes):
        super().__init__(None)
        self.dataset = dataset
        self.batch_size = batch_size
        self.x = dataset.x
        self.y = dataset.y
        self.num_classes = num_classes
        assert batch_size // num_classes * num_classes == batch_size, \
            f'batch size {batch_size} is not a multiple of num of classes {num_classes}'
        print(f'train sampler with batch size = {batch_size} and {num_classes} classes in a batch')
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        indices = list(range(len(self.y)))
        label_cluster = {}
        for i in indices:
            label = self.y[i].item()
            if label not in label_cluster:
                label_cluster[label] = []
            label_cluster[label].append(i)
        for key, value in label_cluster.items():
            random.shuffle(value)

        assert len(label_cluster) > self.num_classes, \
            f'number of available classes {label_cluster} < required classes {self.num_classes}'

        # too time-consuming, i.e., O(|D||C|/|B|)s
        batch_indices = []
        flag = True
        while flag:
            # find candidate classes
            num_samples_per_class = self.batch_size // self.num_classes
            available_classes = list(filter(lambda x: len(label_cluster[x]) >= num_samples_per_class,
                                            list(range(max(self.y) + 1))))
            if len(available_classes) < self.num_classes:
                break
            selected_classes = random.choices(available_classes, k=self.num_classes)
            batch = []
            for c in selected_classes:
                batch.extend(label_cluster[c][-num_samples_per_class:])
                del label_cluster[c][-num_samples_per_class:]
            random.shuffle(batch)
            batch_indices.append(batch)

        random.shuffle(batch_indices)
        # print(batch_indices)

        all = sum(batch_indices, [])

        return iter(all)

    def __len__(self):
        return self.length


class EnsembleDataset(Dataset):
    def __init__(self, x_style, x_char, x_bert, y):
        super(EnsembleDataset, self).__init__()
        self.x_style = x_style
        self.x_char = x_char
        self.x_bert = x_bert
        self.y = y
    
    def __getitem__(self, idx):
        return self.x_style[idx], self.x_char[idx], torch.tensor(self.x_bert['input_ids'][idx]), torch.tensor(self.x_bert['attention_mask'][idx]), self.y[idx]

    def __len__(self):
        return len(self.y)


class TransformerEnsembleDataset(Dataset):
    def __init__(self, x, y, tokenizers, lengths):
        super(TransformerEnsembleDataset, self).__init__()
        self.x = x
        self.tokenizers = tokenizers
        self.lengths = lengths
        self.caches = [{} for i in range(len(tokenizers))]
        self.y = torch.tensor(y)

    def tokenize(self, x, i):
        dic = self.tokenizers[i].batch_encode_plus(
                    batch_text_or_text_pairs=[x],        # input must be a list
                    max_length=self.lengths[i],
                    padding='max_length',
                    truncation=True,
                    return_token_type_ids=True,
                    return_tensors="pt"
                )
        return [x[0] for x in dic.values()]     # get rid of the first dim
    
    def __getitem__(self, idx):
        if idx not in self.caches[0]:
            for i in range(len(self.tokenizers)):
                self.caches[i][idx] = self.tokenize(self.x[idx], i)
        
        return [self.caches[i][idx] for i in range(len(self.tokenizers))], self.y[idx]

    def __len__(self):
        return len(self.y)
