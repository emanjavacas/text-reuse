
import seqmod
from dataiter import lines, DataIter

fname = '/tmp/test.txt'
with open(fname, 'w+') as f:
    for i in range(10000):
        f.write(str(i) + '\n')

d=seqmod.misc.Dict().fit(lines(fname))
batches=DataIter(d, fname, min_len=1).batch_generator(10, buffer_size=1000)
asserts = 0
for (src, _), ((trg1, _), (trg2, _)) in batches:
    for i in range(src.size(1)):
        for s, t1, t2 in zip(src[:, i].data, trg1[:, i].data, trg2[:, i].data):
            s, t1, t2 = int(d.vocab[s]), int(d.vocab[t1]), int(d.vocab[t2])
            assert s == t1 + 1, s == t2 - 1
            asserts += 1

print("Asserts:", asserts)
