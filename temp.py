from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PDPATH import PDPATH

from classes.RNN_Models import Basic_LSTM_Model, Basic_RNN_Model
from classes.Configs import Configs
from reader import Vocab
from classes.Data import TestData, SimpleTestData
import reader
from trainer import run_epoch, get_model
from utilities.test_saver import test_saver

flags = tf.flags
logging = tf.logging
flags.DEFINE_integer("test_type", None, "Specify the type of test to perform")
flags.DEFINE_string("model", None, "Path to trained model.")
flags.DEFINE_string("test", None, "Path to test data.")
flags.DEFINE_string("vocab", None, "Path to vocabulary")

FLAGS = flags.FLAGS


def peek(a):
    print(np.around(a, 2))


def data_type():
    return tf.float32


def load_configs(path):
    for file in os.listdir(path):
        if file.endswith('.config'):
            return pickle.load(open(os.path.join(path, file), 'rb'))


def test_targs(session, model, model_input):
    # Run test epoch
    perp, preds = run_epoch(session, model)

    # Fetch useful information about the test
    targs = model_input.targets.reshape([1, -1])
    meta = model_input.meta
    num_steps = model_input.num_steps
    batch_size = model_input.batch_size
    num_condits = len(meta)
    num_preds = num_steps * batch_size
    max_targs = max([x[1] for x in meta])

    # Create an empty container
    r = np.empty([num_condits, max_targs])
    r[:] = 0

    # Extract useful data
    targ_preds = np.squeeze(preds[np.arange(num_preds), targs])
    j = 0
    for i, (c, ppc) in enumerate(meta):
        crit_inds = [x for x in range(c - 1, num_steps * ppc - 1, num_steps)]
        condit = targ_preds[j:j + num_steps * ppc]
        r[i, 0:ppc] = condit[crit_inds]
        j += num_steps * ppc
    return r


def general_goodness_test(session, model, model_input, pos):
    '''inputs a test set into a model and returns predictions for a given POS'''
    perp, preds = run_epoch(session, model)
    voc = model_input.vocab
    context_size = model_input.num_steps
    stop, _ = np.shape(preds)
    misc_inds = list(range(0, 10000))  # list all voc_size number of inds
    rows = list(range(context_size - 1, stop, context_size))
    preds_given_context = preds[rows, :]
    targ_inds = [x[0] for x in voc.pos2sNid[pos]]
    _ = [misc_inds.remove(x) for x in targ_inds]  # prune out target inds from list of all inds
    targ_preds = preds_given_context[:, targ_inds]
    misc_preds = preds_given_context[:, misc_inds]
    targ_mean = np.mean(targ_preds, 1)
    misc_mean = np.mean(misc_preds, 1)
    goodness = (targ_mean - misc_mean) / misc_mean
    return goodness, preds_given_context


def specific_goodness_test(session, model, model_input, legal_pos, illegal_pos):
    '''inputs a test set into a model and returns predictions for a given POS'''
    voc = model_input.vocab
    perp, preds = run_epoch(session, model)

    context_size = model_input.num_steps
    stop, _ = np.shape(preds)
    rows = list(range(context_size - 1, stop, context_size))
    preds_given_context = preds[rows, :]
    # mean_preds_given_context = np.mean(preds_given_context,1)

    targ_inds = [x[0] for x in voc.pos2sNid[legal_pos]]
    targ_preds = preds_given_context[:, targ_inds]

    illegal_inds = [x[0] for x in voc.pos2sNid[illegal_pos]]
    illegal_preds = preds_given_context[:, illegal_inds]

    targ_mean = np.mean(targ_preds, 1)
    illegal_mean = np.mean(illegal_preds, 1)
    # goodness = targ_mean / (targ_mean + illegal_mean) # Todo try alternative goodness measure e.g.: targ_mean/illegal_mean
    expected_act = 1 / 10000  # <--- add a size feature to the Vocab class for!!
    goodness = targ_mean / (targ_mean + expected_act + illegal_mean)
    return goodness, preds_given_context


def cool_test(session, model, model_input, k, pos1, pos2, vocab):
    '''inputs a test set into a model and returns predictions for a given POS'''
    voc = model_input.vocab
    perp, preds = run_epoch(session, model)
    context_size = model_input.num_steps
    stop, _ = np.shape(preds)
    rows = list(range(context_size - 1, stop, context_size))
    preds_given_context = preds[rows, :]  # <-- row of predictions given context

    targ_inds = [x[0] for x in voc.pos2sNid[pos1]]
    targ_preds = preds_given_context[:, targ_inds]  # <-- select only target predictions
    illegal_inds = [x[0] for x in voc.pos2sNid[pos2]]
    illegal_preds = preds_given_context[:, illegal_inds]  # <-- select only illegal predictions

    top_overall = session.run(tf.nn.top_k(input=preds_given_context, k=k, sorted=True)).indices
    top_targets = session.run(tf.nn.top_k(input=targ_preds, k=k, sorted=True)).indices
    top_illegal = session.run(tf.nn.top_k(input=illegal_preds, k=k, sorted=True)).indices

    top_acts = []
    top_words = []
    top_inds = []
    for i, (overall, target, illegal) in enumerate(zip(top_overall, top_targets, top_illegal)):
        topa, topw, topi = [], [], []
        topa.append([preds_given_context[i, ind] for ind in overall])
        topa.append([targ_preds[i, ind] for ind in target])
        topa.append([illegal_preds[i, ind] for ind in illegal])

        topw.append([vocab.gets(ind) for ind in overall])
        topw.append([vocab.gets(targ_inds[ind]) for ind in target])
        topw.append([vocab.gets(illegal_inds[ind]) for ind in illegal])

        topi.append([vocab.getid(w) for w in topw[0]])
        topi.append([vocab.getid(w) for w in topw[1]])
        topi.append([vocab.getid(w) for w in topw[2]])

        top_acts.append(topa)
        top_words.append(topw)
        top_inds.append(topi)

    return top_acts, top_words, top_inds, np.mean(targ_preds, 1), np.mean(illegal_preds, 1)


def top_choices(session, k, preds, vocab):
    top_inds = session.run(tf.nn.top_k(input=preds, k=k, sorted=True)).indices
    top_words = []
    for row in top_inds:
        top_words.append([vocab.gets(ind) for ind in row])
    return top_inds, top_words


def main(_):
    # TEST 1
    # ================================================================================================================
    if FLAGS.test_type == 1:
        vocab = reader.get_vocab(FLAGS.vocab)
        test_ids, test_meta = reader.make_test(PDPATH('/test_data/' + FLAGS.test), vocab)
        model_path = PDPATH('/trained_models/') + FLAGS.model
        config = load_configs(model_path)

        with tf.Graph().as_default() as graph:
            with tf.Session() as session:
                test_input = TestData(config=config,
                                      test_data=test_ids,
                                      test_meta=test_meta,
                                      vocab=vocab,
                                      name="TestInput")

                with tf.variable_scope("Model"):
                    mtest = get_model(cell=config.model, is_training=False, config=config, input_=test_input)

                saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
                saved_files = os.listdir(model_path)
                for file in saved_files:
                    if '.meta' in file:
                        ckpt = file.split(sep='.')[0]
                        saver.restore(session, os.path.join(model_path, ckpt))
                        continue

                np.set_printoptions(precision=4, suppress=False, linewidth=100)
                b = test_targs(session=session, model=mtest, model_input=test_input)
                print(b)
                b = b / np.sum(b, axis=1).reshape([-1, 1])

                np.set_printoptions(precision=4, suppress=False, linewidth=100)
                print(b)

    # TEST 2
    # ================================================================================================================
    elif FLAGS.test_type == 2:
        vocab = reader.get_vocab(FLAGS.vocab)
        test_ids = reader._file_to_word_ids(PDPATH('/test_data/' + FLAGS.test), vocab.s2id)
        model_path = PDPATH('/trained_models/') + FLAGS.model
        config = load_configs(model_path)

        targ_num = FLAGS.test.split('.')[0].split('_')[1].split('-')[0]  # parse test name to extract useful info\
        if targ_num == 's':
            targ_pos, illegal = 'VBZ', 'VBP'
        else:
            targ_pos, illegal = 'VBP', 'VBZ'

        with tf.Graph().as_default() as graph:
            with tf.Session() as session:
                test_input = SimpleTestData(config=config,
                                            test_data=test_ids,
                                            vocab=vocab,
                                            name="TestInput")

                with tf.variable_scope("Model"):
                    mtest = get_model(cell=config.model, is_training=False, config=config, input_=test_input)

                saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
                saved_files = os.listdir(model_path)
                for file in saved_files:
                    if '.meta' in file:
                        ckpt = file.split(sep='.')[0]
                        saver.restore(session, os.path.join(model_path, ckpt))
                        continue

                gscores, preds = specific_goodness_test(session=session,
                                                        model=mtest,
                                                        model_input=test_input,
                                                        legal_pos=targ_pos,
                                                        illegal_pos=illegal)

                top10_inds, top10_words = top_choices(session=session, k=10, preds=preds, vocab=vocab)

        print('{} ({}):  {}'.format(FLAGS.model.split('_')[0] + '_' + FLAGS.model.split('_')[-1],
                                    FLAGS.test.split('.')[0], np.round(np.mean(gscores), 4)))

        test_saver(save_to='results.csv',
                   model=FLAGS.model,
                   test=FLAGS.test.split('.')[0],
                   scores=gscores,
                   top_inds=top10_inds,
                   top_words=top10_words)

    # TEST 3
    # ================================================================================================================
    elif FLAGS.test_type == 3:
        vocab = reader.get_vocab(FLAGS.vocab)
        test_ids = reader._file_to_word_ids(PDPATH('/test_data/' + FLAGS.test), vocab.s2id)
        model_path = PDPATH('/trained_models/') + FLAGS.model
        config = load_configs(model_path)

        with tf.Graph().as_default() as graph:
            with tf.Session() as session:

                test_input = SimpleTestData(config=config,
                                            test_data=test_ids,
                                            vocab=vocab,
                                            name="TestInput")

                with tf.variable_scope("Model"):
                    mtest = get_model(cell=config.model, is_training=False, config=config, input_=test_input)

                saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
                saved_files = os.listdir(model_path)
                for file in saved_files:
                    if '.meta' in file:
                        ckpt = file.split(sep='.')[0]
                        saver.restore(session, os.path.join(model_path, ckpt))
                        continue

                top_acts, top_words, top_inds, mt, mi = cool_test(session=session,
                                                                  model=mtest,
                                                                  model_input=test_input,
                                                                  k=5,
                                                                  pos1='VBZ',
                                                                  pos2='VBP',
                                                                  vocab=vocab)

        # for qqq in top_acts:
        #     for qq in qqq:
        #         print(qq)
        new_file = {}
        with open(PDPATH('/test_data/' + FLAGS.test), 'r') as file:
            for i, line in enumerate(file.readlines()[:-1]):
                print(i)
                new_file[line] = {'xtop': top_inds[i], 'xbot': top_words[i], 'vals': top_acts[i]}
        pickle.dump(new_file, open(config.model + '_' + FLAGS.test.split('.')[0], 'wb'))

        # print('{} ({}):  {}'.format(FLAGS.model.split('_')[0] + '_' + FLAGS.model.split('_')[-1], FLAGS.test.split('.')[0],
        #                             np.round(np.mean(gscores), 4)))

        # test_saver(save_to='results.csv',
        #            model=FLAGS.model,
        #            test=FLAGS.test.split('.')[0],
        #            scores=gscores,
        #            top_inds=top10_inds,
        #            top_words=top10_words)


if __name__ == "__main__": tf.app.run()