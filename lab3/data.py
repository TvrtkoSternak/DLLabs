import numpy as np
from collections import Counter


class DataLoader:
    sequence_length = 0
    batch_size = 0
    batch_pointer = 0
    char2id = dict()
    id2char = {}
    x = None
    minibatches_x = list()
    minibatches_y = list()
    num_batches = 0
    sorted_chars = Counter()

    def __init__(self, input_file, batch_size, sequence_length):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.preprocess(input_file)
        self.create_minibatches()
        pass

    def preprocess(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            data = f.read()  # python 2

        # count and sort most frequent characters

        for x in data:
            self.sorted_chars += Counter(x.strip())

        most_common_list = list()

        for x,_ in self.sorted_chars.most_common():
            most_common_list.append(x)

        # self.sorted chars contains just the characters ordered descending by frequency
        self.char2id = dict(zip(most_common_list, range(len(most_common_list))))

        # reverse the mapping
        self.id2char = {k: v for v, k in self.char2id.items()}
        # convert the data to ids
        self.x = np.array(list(map(self.char2id.get, data)))

    def encode(self, sequence):
        # returns the sequence encoded as integers
        pass

    def decode(self, encoded_sequence):
        # returns the sequence decoded as letters
        pass

    def create_minibatches(self):
        self.num_batches = int(
            len(self.x) / (self.batch_size * self.sequence_length))  # calculate the number of batches

        # Is all the data going to be present in the batches? Why?
        # What happens if we select a batch size and sequence length larger than the length of the data?

        #######################################
        #       Convert data to batches       #
        #######################################

        pass

    def next_minibatch(self):
        # ...

        batch_x, batch_y = None, None
        # handling batch pointer & reset
        # new_epoch is a boolean indicating if the batch pointer was reset
        # in this function call

        batch_x = self.minibatches_x[self.batch_pointer]
        batch_y = self.minibatches_y[self.batch_pointer]

        new_epoch = self.batch_pointer == 0
        self.batch_pointer = (self.batch_pointer + 1) % len(self.minibatches_x)

        return new_epoch, batch_x, batch_y

if __name__ == "__main__":
    DataLoader('selected_text.txt', 10, 10)