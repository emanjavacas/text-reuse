
import operator
import random
import itertools

import torch
from seqmod.modules.torch_utils import flip


def chunks(it, size):
    """
    Chunk a generator into a given size (last chunk might be smaller)
    """
    buf = []
    for s in it:
        buf.append(s)
        if len(buf) == size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def lines(*paths, min_len=1, max_len=-1, verbose=True):
    """
    Generator over lines from files tokenized and length-processed
    """
    for idx, path in enumerate(paths):
        with open(path, 'r') as f:
            try:
                num = 1
                for line in f:
                    num += 1
                    line = line.strip().split()
                    if len(line) < min_len:
                        yield None
                    elif (max_len > 0 and len(line) > max_len):
                        yield line[:max_len]  # crop
                    else:
                        yield line
            except UnicodeDecodeError:
                if verbose:
                    print("[{}]: Read error at line {}".format(path, num))
                yield None


def window(it):
    """
    >>> list(window(range(10)))
    [(None, 0, 1), (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, None)]
    """
    it = itertools.chain([None], it, [None])  # pad for completeness
    result = tuple(itertools.islice(it, 3))

    if len(result) == 3:
        yield result

    for elem in it:
        result = result[1:] + (elem,)
        yield result


class DataIter(object):

    # - larger chunks => more pairs (less split points => more subsequent sents)
    # - since sents are pooled (see `lines`), false pairs exist between files
    # - selecting both contexts results in less pairs

    def __init__(self, d, *paths, gpu=False, includes=(True, True),
                 min_len=3, max_len=50, shuffle=True, always_reverse=False):
        self.d = d
        self.paths = list(paths)
        self.gpu = gpu
        self.prevline, self.nextline = includes
        self.min_len = min_len
        self.max_len = max_len
        self.shuffle = shuffle
        self.always_reverse = always_reverse

    def wrap(self, data, reverse=False):
        data = list(self.d.transform(data))
        data, lengths = self.d.pack(
            data, return_lengths=True, align_right=reverse)
        data, lengths = torch.autograd.Variable(data), torch.LongTensor(lengths)

        if reverse:
            data = flip(data, dim=0)  # [<eos> ... <bos> <pad> <pad>]

        if self.gpu:
            data, lengths = data.cuda(), lengths.cuda()

        return data, lengths

    def pack_batch(self, batch):
        inp, sents = zip(*batch)
        inp = self.wrap(inp)

        if self.prevline and self.nextline:
            prevline, nextline = zip(*sents)
            sents = (self.wrap(prevline, reverse=self.always_reverse),
                     self.wrap(nextline, reverse=True))
        elif self.prevline:
            sents = (self.wrap(sents, reverse=self.always_reverse), None)
        else:
            sents = (None, self.wrap(sents, reverse=True))

        return inp, sents

    def batches(self, buf, batch_size):
        def key(tup):  # sort by target lengths
            if self.prevline and self.nextline:
                return len(tup[1][0])
            else:
                return len(tup[1])

        buf, batches = sorted(buf, key=key), []

        while len(buf) >= batch_size:
            batches.append(self.pack_batch(buf[:batch_size]))
            buf = buf[batch_size:]

        if self.shuffle:
            random.shuffle(batches)

        return batches, buf

    def batch_generator(self, batch_size, buffer_size=10000):
        if batch_size > buffer_size:
            raise ValueError("`batch_size` can't be larger than"
                             " buffer capacity {}".format(buffer_size))

        random.shuffle(self.paths)

        lines_it = lines(*self.paths, min_len=self.min_len, max_len=self.max_len)
        buf = []
        for chunk in chunks(lines_it, buffer_size):
            for (prevline, current, nextline) in window(chunk):
                if current is not None:
                    if self.prevline and self.nextline:
                        if prevline is not None and nextline is not None:
                            buf.append((current, (prevline, nextline)))
                    elif self.prevline and prevline is not None:
                        buf.append((current, prevline))
                    elif self.nextline and nextline is not None:
                        buf.append((current, nextline))
            batches, buf = self.batches(buf, batch_size)
            for batch in batches:
                yield batch

        batches, last = self.batches(buf, batch_size)
        for batch in batches:
            yield batch
        if len(last) > 0:
            yield self.pack_batch(last)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+')
    parser.add_argument('--action')
    parser.add_argument('--dict_path')
    parser.add_argument('--max_size', type=int, default=20000)
    parser.add_argument('--buffer_size', type=int, default=10000)
    args = parser.parse_args()

    import seqmod
    import glob

    min_len, max_len = 3, 35
    if args.action == 'dict':
        D = seqmod.misc.dataset.Dict(
            pad_token=seqmod.utils.PAD, bos_token=seqmod.utils.BOS,
            eos_token=seqmod.utils.EOS, max_size=args.max_size
        ).fit((line for line in lines(*args.files, min_len=min_len, max_len=max_len)
               if line is not None))
        seqmod.utils.save_model(
            D, '{}.vocab{}.dict'.format(args.dict_path, args.max_size))

    elif args.action == 'create_sample_files':

        def create_sample_files(files=10):
            import lorem
            
            for i in range(files):
                with open('lorem{}.txt'.format(i+1), 'w+') as f:
                    for _ in range(random.randint(500, 1500)):
                        f.write('{}\n'.format(lorem.sentence()))

        create_sample_files(files=500)

    else:
        if args.dict_path:
            D = seqmod.load_model(args.dict_path)
        else:
            D = seqmod.dataset.Dict(
                pad_token=seqmod.utils.PAD, bos_token=seqmod.utils.BOS,
                eos_token=seqmod.utils.EOS
            ).fit(lines(*args.files, min_len=min_len, max_len=max_len))
        dataiter = DataIter(D, *args.files, max_len=max_len, min_len=min_len)
    
        import time
        # start buffer
        batches = dataiter.batch_generator(
            100, buffer_size=args.buffer_size, shuffle=False)
        start, sents, speed = time.time(), 0, []
        # start batches
        for batch in batches:
            (inp, lengths), _ = batch
            sents += len(lengths)
            restart = time.time()
            speed.append(restart - start)
            start = restart

        from statistics import mean
        print("Sents: {}; speed: {:.3f} msec/batch; buffer: {}".format(
            sents, mean(speed) * 1000, args.buffer_size))

