
import os
import glob

from seqmod.modules.embedding import Embedding
from seqmod.misc import Checkpoint
import seqmod.utils as u

from text_reuse.skipthought.dataiter import DataIter

from .skipthought import SkipThoughts, Loss


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--paths', required=True)
    parser.add_argument('--dict_path', required=True)
    parser.add_argument('--min_len', type=int, default=2)
    parser.add_argument('--max_len', type=int, default=35)
    parser.add_argument('--lower', action='store_true')
    parser.add_argument('--num', action='store_true')
    parser.add_argument('--level', default='word')
    # model
    parser.add_argument('--mode', default='double')
    parser.add_argument('--softmax', default='single')
    parser.add_argument('--emb_dim', type=int, default=620)
    parser.add_argument('--hid_dim', type=int, default=2400)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--cell', default='GRU')
    parser.add_argument('--summary', default='last')
    parser.add_argument('--train_init', action='store_true')
    parser.add_argument('--init_embeddings', action='store_true')
    parser.add_argument('--embeddings_path',
                        default='/home/corpora/word_embeddings/' +
                        'glove.840B.300d.txt')
    # training
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_schedule_checkpoints', type=int, default=1)
    parser.add_argument('--lr_schedule_factor', type=float, default=1)
    parser.add_argument('--max_norm', type=float, default=5.)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--buffer_size', type=int, default=int(5e+6))
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--save_freq', type=int, default=50000)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    embeddings = Embedding.from_dict(u.load_model(args.dict_path), args.emb_dim)

    checkpoint = None
    if not args.test:
        modeldir = 'SkipThoughts-{}'.format(args.mode)
        checkpoint = Checkpoint(modeldir, keep=5).setup(args)

    m = SkipThoughts(embeddings, args.mode, cell=args.cell, hid_dim=args.hid_dim,
                     num_layers=args.num_layers, summary=args.summary,
                     softmax=args.softmax, dropout=args.dropout,
                     max_norm=args.max_norm, optimizer=args.optimizer, lr=args.lr,
                     save_freq=args.save_freq, checkpoint=checkpoint)

    print(m)
    print()
    print("Parameter stats ...\n")
    m.report_params()

    print()
    print("Initializing parameters ...\n")
    u.initialize_model(
        m,
        rnn={'type': 'rnn_orthogonal', 'args': {'forget_bias': True}},
        emb={'type': 'uniform', 'args': {'a': -0.1, 'b': 0.1}})

    if args.init_embeddings:
        embeddings.init_embeddings_from_file(args.embeddings_path, verbose=True)

    if args.gpu:
        m.cuda()

    m.train()

    paths = glob.glob(os.path.expanduser(args.paths))

    dataiter = DataIter(m.encoder.embeddings.d, *paths, includes=Loss.MODES[args.mode],
                        min_len=args.min_len, max_len=args.max_len, gpu=args.gpu,
                        always_reverse=args.mode=='clone')

    print()
    print("Starting training ...\n")
    try:
        for epoch in range(1, args.epochs + 1):
            print("***{} Epoch #{} {}***".format("---" * 4, epoch, "---" * 4))
            m.train_model(dataiter.batch_generator(args.batch_size,
                                                   buffer_size=args.buffer_size))
            print()
    except KeyboardInterrupt:
        print("Bye!")
    finally:
        print("Finished!")
