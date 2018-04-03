

def get_targets(vocab, *pairs):
    targets = []
    vocab = set(vocab)

    for ps in pairs:
        for (p1, p2), _ in ps:
            for w in p1 + p2:
                if w not in vocab:
                    targets.append(w)

    return list(set(targets))


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


def get_line(line, min_len, max_len):
    """
    Processed a line wrt length. Returns None if line is not conformant
    """
    line = line.strip().split()
    if len(line) < min_len or len(line) > max_len:
        return None
    return line


def lines(*paths, min_len, max_len):
    """
    Generator over lines from files tokenized and length-processed
    """
    for path in paths:
        with open(path, 'r') as f:
            for line in f:
                yield get_line(line, min_len, max_len)


def window(it):
    """
    >>> list(window(range(10)))
    [(None, 0, 1), (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, None)]
    """
    it = itertools.chain([None], it, [None])
    result = tuple(itertools.islice(it, 3))

    if len(result) == 3:
        yield result

    for elem in it:
        result = result[1:] + (elem,)
        yield result


class Iterator(object):
    def __init__(self, d, *paths, mode=(True, True), max_len=50, buffer_size=1000):
        self.d = d
        self.paths = paths
        self.mode = mode
        self.max_len = max_len
        self.buffer_size = buffer_size
        self.buf = []

    def batches(self, buf):
        split, rest = [], []
        return split, rest

    def batch_generator(self, batch_size, shuffle=True):
        if batch_size > self.buffer_size:
            raise ValueError("`batch_size` can't be larger than buffer "
                             "capacity {}".format(self.buffer_size))

        if shuffle:
            random.shuffle(self.paths)

        buf, prevline, nextline = [], None, None
        for chunk in chunks(lines(*self.paths), self.buffer_size):
            prevline = get_line(next(chunk), 3, self.max_len)
            currline = get_line(next(chunk), 3, self.max_len)
            for nextline in chunk:
                nextline = get_line(nextline, 3, self.max_len)
                if nextline is None:
                    continue

            batches, buf = self.batches(buf)
            for batch in batches:
                yield batch

        batches, last = self.batches(buf)
        for batch in batches:
            yield batch
        yield last
