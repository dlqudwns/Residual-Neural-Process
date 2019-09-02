import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime, os
import pandas as pd
import time

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    logdir = FLAGS.logdir and os.path.expanduser(os.path.join(
        FLAGS.logdir, '{}-{}-{}-{}'.format(
            FLAGS.timestamp, FLAGS.exp, FLAGS.model, FLAGS.pid)))
    print(logdir)

    if FLAGS.exp in ['mnist', 'celeba']:
        num_epoch = 500
        batch_size = 16
        z_dim = 250

        PLOT_AFTER = int(4000)
        tf.reset_default_graph()

        from datasets.image2d import ImagesReader
        dataset = ImagesReader(
            batch_size=batch_size, dataset_name=FLAGS.exp, min_num_context=3
            ,include_context = False, testing_only=(FLAGS.loaddir is not None))
        data_train = dataset.generate_data(testing=False)
        cx, cy, tx, ty = dataset.generate_data(testing=True)

        from models.image2d_stochastic import ResidualNeuralProcess
        model = ResidualNeuralProcess(FLAGS.model, data_train, z_dim, dy=dataset.dy)
        print('Model : toy')
    else:
        raise NotImplementedError

    saver = tf.train.Saver()
    NLLs = []; times = []; global_steps = []

    try:
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(str(logdir), sess.graph)
            tf.global_variables_initializer().run()
            dataset.initialize(sess)

            if FLAGS.loaddir is not None:
                from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
                checkpoint = tf.train.latest_checkpoint(os.path.expanduser(os.path.join(
                    FLAGS.logdir, '{}-{}-{}-{}'.format(
                        FLAGS.loaddir, FLAGS.exp, FLAGS.model, FLAGS.pid))))
                saver.restore(sess, checkpoint)
                print("Model restored - ", checkpoint)
                print_tensors_in_checkpoint_file(checkpoint, all_tensors=True, tensor_name='')

            global_step = 0
            merged_summary_op = tf.summary.merge_all()
            for epoch in range(num_epoch):
                # Train
                if FLAGS.loaddir is None:
                    nll_sum = 0; ELBO_sum = 0; ZKL_sum = 0; FKL_sum = 0
                    for i in tqdm(range(PLOT_AFTER), desc='[Epoch {}/{}]'.format(epoch, num_epoch)):
                        sess.run(model.optims)
                        global_step += 1
                        if i % 20 == 0:
                            nll, elbo, zkl, fkl = sess.run([model.NLL, model.ELBO, model.ZKL, model.FKL])
                            nll_sum += nll;ELBO_sum += elbo;ZKL_sum += zkl;FKL_sum += fkl
                    nll_mean, elbo_mean, zkl_mean, fkl_mean = \
                        [x / PLOT_AFTER * 20 for x in [nll_sum, ELBO_sum, ZKL_sum, FKL_sum]]

                    NLLs.append(nll_mean)
                    times.append(time.time());global_steps.append(global_step)

                    nll_summary = tf.Summary()
                    nll_summary.value.add(tag='NLL', simple_value=nll_mean)
                    nll_summary.value.add(tag='ELBO', simple_value=elbo_mean)
                    nll_summary.value.add(tag='ZKL', simple_value=zkl_mean)
                    nll_summary.value.add(tag='FKL', simple_value=fkl_mean)
                    writer.add_summary(nll_summary, global_step)

                    summary = sess.run(merged_summary_op)
                    writer.add_summary(summary, global_step)

                if FLAGS.loaddir is not None:
                    import sys
                    sys.exit()
                print('[Epoch {}] Save Parameters'.format(epoch))
                saver.save(sess, str(logdir + '/model'), global_step=global_step)
    finally:
        df = pd.DataFrame({'NLL':NLLs, 'time':times, 'global_step':global_steps})
        with open(logdir + '.csv', 'w') as f:
            df.to_csv(f, header=False)

def plot_functions(plots, save_dir):
    plt.clf()
    pys = [py[:, 0, :] for py in plots[:4]]
    cxys = [cy[0, :] for cy in plots[4:]]
    tx, ty = cxys[-2], cxys[-1]
    cxys = cxys[:-2]

    fig = plt.figure(figsize=(7,8))
    ax = []
    rows = 8
    cols = 7
    shape = [28, 28] if FLAGS.exp == 'mnist' else [32, 32, 3]
    cmap = 'gray' if FLAGS.exp == 'mnist' else None

    for j in range(4):
        ax.append(fig.add_subplot(rows, cols, 1 + 14 * j))
        context_img = np.zeros(shape)
        if FLAGS.exp == 'mnist':
            context_img = np.zeros(shape + [3])
            context_img[:,:,2] = 1.0
        cx, cy = cxys[j*2], cxys[j*2+1]
        for x, y in zip(cx, cy):
            x = np.round((x + 1) * shape[0] / 2).astype(np.int32)
            if FLAGS.exp == 'mnist':
                context_img[x[0], x[1]] = [y + 0.5] * 3
            else:
                context_img[x[0], x[1]] = y + 0.5

        plt.imshow(context_img, cmap=None, vmin=0.0, vmax=1.0)
        plt.tick_params(axis='both', which='both',
                        bottom=False, left=False, labelbottom=False, labelleft=False)

        ax.append(fig.add_subplot(rows, cols, 2 + 14 * j))
        target_img = np.zeros(shape)
        for x, y in zip(tx, ty):
            x = np.round((x + 1) * shape[0] / 2).astype(np.int32)
            target_img[x[0], x[1]] = y + 0.5
        plt.imshow(target_img, cmap=cmap, vmin=0.0, vmax=1.0)
        plt.tick_params(axis='both', which='both',
                        bottom=False, left=False, labelbottom=False, labelleft=False)

        for i in range(10):
            offset = 3 if i < 5 else 5
            ax.append(fig.add_subplot(rows, cols, i + offset + 14 * j))
            target_img = np.zeros(shape)
            for x, y in zip(tx, pys[j][i]):
                x = np.round((x + 1) * shape[0] / 2).astype(np.int32)
                target_img[x[0], x[1]] = y + 0.5
            plt.imshow(target_img, cmap=cmap, vmin=0.0, vmax=1.0)
            plt.tick_params(axis='both', which='both',
                            bottom=False, left=False, labelbottom=False, labelleft=False)
        #for a in ax:
        #    a.axis('off')
    ax[0].set_xlabel('Context')
    ax[0].xaxis.set_label_position('top')
    ax[1].set_xlabel('Ground Truth')
    ax[1].xaxis.set_label_position('top')
    ax[2].set_xlabel('Samples')
    ax[2].xaxis.set_label_position('top')

    ax[0].set_ylabel('10 points')
    ax[12].set_ylabel('30 points')
    ax[24].set_ylabel('100 points')
    ax[36].set_ylabel('Upper half')

    if FLAGS.model == 'ANP':
        title = 'Attentive NP'
    elif FLAGS.model == 'RNP':
        title = 'Residual NP (stochastic path)'
    elif FLAGS.model == 'bRNP':
        title = 'Residual NP (both stochasity)'
    else:
        title = 'Bayesian Last Layer'

    # Make the plot pretty
    plt.grid('off')
    ax = plt.gca()
    ax.set_facecolor('white')
    plt.tight_layout()
    plt.suptitle(title)
    fig.subplots_adjust(hspace=0, wspace=0, top=0.93)
    plt.savefig(save_dir, format='pdf')


if __name__ == '__main__':
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string(
        'logdir', '/ext/bjlee/logs3/',
        'Base directory to store logs.')
    tf.app.flags.DEFINE_string(
        'loaddir', None,
        'Base directory to load model')
    tf.app.flags.DEFINE_string(
        'timestamp', datetime.datetime.now().strftime('%Y%m%dT%H%M%S'),
        'Sub directory to store logs.')
    tf.app.flags.DEFINE_string(
        'exp', None,
        'Experiment to execute.')
    tf.app.flags.DEFINE_string(
        'model', 'ANP',
        'Experiment to execute.')
    tf.app.flags.DEFINE_string(
        'pid', '',
        'additional string')
    tf.app.run()
