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
        FLAGS.logdir, '{}-toy-{}-{}-{}'.format(
            FLAGS.timestamp, FLAGS.model, FLAGS.zdim, FLAGS.pid)))
    print(logdir)

    num_epoch = 500
    batch_size = 16
    z_dim = int(FLAGS.zdim)

    MAX_CONTEXT_POINTS = 100
    PLOT_AFTER = int(4000)
    tf.reset_default_graph()

    from datasets.toy1d import GPCurvesReader
    # Train dataset
    dataset_train = GPCurvesReader(
        batch_size=batch_size, max_num_context=MAX_CONTEXT_POINTS,
        include_context=FLAGS.include_context)
    data_train = dataset_train.generate_curves()

    # Test dataset
    dataset_test = GPCurvesReader(
        batch_size=1, max_num_context=MAX_CONTEXT_POINTS, testing=True,
        include_context=FLAGS.include_context)
    data_test = dataset_test.generate_curves()

    from models.toy1d import ResidualNeuralProcess
    model = ResidualNeuralProcess(
        FLAGS.model, data_train, z_dim, is_feature_shared=FLAGS.is_feature_shared)
    test_mean, test_sigma = model.predict(data_test, 20)
    print('Model : toy')

    var_list = [t for t in tf.trainable_variables() if 'sigma' not in t.name]
    saver = tf.train.Saver(var_list=var_list)
    cNLLs = []; NLLs = []; times = []; global_steps = []

    try:
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(str(logdir), sess.graph)
            tf.global_variables_initializer().run()

            if FLAGS.loaddir is not None:
                from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
                checkpoint = tf.train.latest_checkpoint(os.path.expanduser(os.path.join(
                        FLAGS.logdir, FLAGS.loaddir)))
                saver.restore(sess, checkpoint)
                print("Model restored - ", checkpoint)
                print_tensors_in_checkpoint_file(checkpoint, all_tensors=True, tensor_name='')

            global_step = 0
            for epoch in range(num_epoch):
                # Train
                if FLAGS.loaddir is None:
                    nll_sum = 0; cnll_sum = 0
                    for _ in tqdm(range(PLOT_AFTER), desc='[Epoch {}/{}]'.format(epoch, num_epoch)):
                        _, nll, cnll = sess.run([model.optims, model.NLL, model.context_NLL])
                        global_step += 1; nll_sum += nll; cnll_sum += cnll
                    nll_mean, cnll_mean = nll_sum / PLOT_AFTER, cnll_sum / PLOT_AFTER
                    cNLLs.append(cnll_mean);NLLs.append(nll_mean)
                    times.append(time.time());global_steps.append(global_step)

                    nll_summary = tf.Summary()
                    nll_summary.value.add(tag='NLL', simple_value=nll_mean)
                    nll_summary.value.add(tag='cNLL', simple_value=cnll_mean)
                    writer.add_summary(nll_summary, global_step)

                # Validation
                pred_y, sigma_y, tcx, tcy, ttx, tty \
                    = sess.run([test_mean, test_sigma] + list(data_test))
                plot_functions(
                    ttx, tty, tcx, tcy, pred_y, sigma_y, '{}/{}.pdf'.format(logdir, global_step))
                if FLAGS.loaddir is not None:
                    import sys
                    sys.exit()
                print('[Epoch {}] Save Parameters'.format(epoch))
                saver.save(sess, str(logdir + '/model'), global_step=global_step)
    finally:
        df = pd.DataFrame({'cNLL':cNLLs,'NLL':NLLs, 'time':times, 'global_step':global_steps})
        with open(logdir + '.csv', 'w') as f:
            df.to_csv(f, header=False)

def plot_functions(target_x, target_y, context_x, context_y, pred_y, sigma_y, save_dir):
    tx, ty, cx, cy = target_x[0, :, 0], target_y[0, :, 0], context_x[0, :, 0], context_y[0, :, 0]
    print(cx)
    print(cy)
    plt.figure(figsize=(3.5,2.5))
    if np.ndim(pred_y) == 4:
        py, sy = pred_y[:, 0, :, 0], sigma_y[:, 0, :, 0]
        for pys, sys in zip(py, sy):
            plt.plot(tx, pys, 'b', linewidth=2, alpha=0.02)
            plt.fill_between(
                tx, pys - sys, pys + sys, alpha=0.02, facecolor='#65c9f7', interpolate=True)
    else:
        py, sy = pred_y[0, :, 0], sigma_y[0, :, 0]
        plt.fill_between(tx, py - sy, py + sy, alpha=0.4, facecolor='#65c9f7', interpolate=True)
        plt.plot(tx, py, 'b', linewidth=2, alpha=0.4)
    #plt.plot(tx, ty, 'k:', linewidth=2)
    plt.plot(cx, cy, 'ko', marker='x')

    if FLAGS.model == 'cANP':
        title = 'Attentive NP'
    elif FLAGS.model == 'RNP_var':
        title = 'Residual NP (var)'
    elif FLAGS.model == 'RNP_full':
        title = 'Residual NP (full)'
    else:
        title = 'Bayesian Last Layer'
    plt.title(title)

    # Make the plot pretty
    plt.yticks([-2.5, 0, 2.5], fontsize=16)
    plt.xticks([-20, 0, 20], fontsize=16)
    plt.ylim([-2.5, 3.5])
    frame = plt.gca()
    frame.axes.xaxis.set_ticklabels([])
    frame.axes.yaxis.set_ticklabels([])
    plt.grid('off')
    ax = plt.gca()
    ax.set_facecolor('white')
    plt.tight_layout()
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
        'model', 'RNP_var',
        'Experiment to execute.')
    tf.app.flags.DEFINE_string(
        'pid', '',
        'additional string')
    tf.app.flags.DEFINE_string(
        'zdim', '150',
        'z dimension')
    tf.app.flags.DEFINE_bool(
        'include_context', False,
        'Should context be included in target')
    tf.app.flags.DEFINE_bool(
        'is_feature_shared', True,
        'is feature shared')
    tf.app.run()
