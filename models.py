import tensorflow as tf
import tensorflow.contrib.slim as slim


def build_pred(x_in, H, phase):
    '''
    This function builds the prediction model
    '''
    num_class = H['num_class']

    conv_kernel_1 = [1, 1]
    conv_kernel_3 = [3, 3]
    pool_kernel = [2, 2]
    pool_stride = 2

    early_feature = {}
    reuse = {'train': False, 'validate': True, 'test': False}[phase]

    with slim.arg_scope(argument_scope(H, phase)):
        scope_name = 'block_1'
        x_input = x_in
        num_outputs = 64
        with tf.variable_scope(scope_name, reuse=reuse):
            layer_1 = slim.conv2d(x_input, num_outputs, conv_kernel_3, scope='conv1')
            layer_2 = slim.conv2d(layer_1, num_outputs, conv_kernel_3, scope='conv2')
            early_feature[scope_name] = layer_2

        scope_name = 'block_2'
        x_input = slim.max_pool2d(layer_2)
        num_outputs = 128
        with tf.variable_scope(scope_name, reuse=reuse):
            layer_1 = slim.conv2d(x_input, num_outputs, conv_kernel_3, scope='conv1')
            layer_2 = slim.conv2d(layer_1, num_outputs, conv_kernel_3, scope='conv2')
            early_feature[scope_name] = layer_2

        scope_name = 'block_3'
        x_input = slim.max_pool2d(layer_2)
        num_outputs = 256
        with tf.variable_scope(scope_name, reuse=reuse):
            layer_1 = slim.conv2d(x_input, num_outputs, conv_kernel_3, scope='conv1')
            layer_2 = slim.conv2d(layer_1, num_outputs, conv_kernel_3, scope='conv2')
            early_feature[scope_name] = layer_2

        scope_name = 'block_4'
        x_input = slim.max_pool2d(layer_2)
        num_outputs = 512
        with tf.variable_scope(scope_name, reuse=reuse):
            layer_1 = slim.conv2d(x_input, num_outputs, conv_kernel_3, scope='conv1')
            layer_2 = slim.conv2d(layer_1, num_outputs, conv_kernel_3, scope='conv2')
            early_feature[scope_name] = layer_2

        scope_name = 'block_5'
        x_input = slim.max_pool2d(layer_2)
        num_outputs = 1024
        with tf.variable_scope(scope_name, reuse=reuse):
            layer_1 = slim.conv2d(x_input, num_outputs, conv_kernel_3, scope='conv1')
            layer_2 = slim.conv2d(layer_1, num_outputs, conv_kernel_3, scope='conv2')
            early_feature[scope_name] = layer_2

        scope_name = 'block_6'
        num_outputs = 512
        with tf.variable_scope(scope_name, reuse=reuse):
            trans_layer = slim.conv2d_transpose(
                layer_2, num_outputs, pool_kernel, pool_stride, scope='conv_trans')
            x_input = tf.concat([early_feature['block_4'], trans_layer], axis=3)
            layer_1 = slim.conv2d(x_input, num_outputs, conv_kernel_3, scope='conv1')
            layer_2 = slim.conv2d(layer_1, num_outputs, conv_kernel_3, scope='conv2')
            early_feature[scope_name] = layer_2

        scope_name = 'block_7'
        num_outputs = 256
        with tf.variable_scope(scope_name, reuse=reuse):
            trans_layer = slim.conv2d_transpose(
                layer_2, num_outputs, pool_kernel, pool_stride, scope='conv_trans')
            x_input = tf.concat([early_feature['block_3'], trans_layer], axis=3)
            layer_1 = slim.conv2d(x_input, num_outputs, conv_kernel_3, scope='conv1')
            layer_2 = slim.conv2d(layer_1, num_outputs, conv_kernel_3, scope='conv2')
            early_feature[scope_name] = layer_2

        scope_name = 'block_8'
        num_outputs = 128
        with tf.variable_scope(scope_name, reuse=reuse):
            trans_layer = slim.conv2d_transpose(
                layer_2, num_outputs, pool_kernel, pool_stride, scope='conv_trans')
            x_input = tf.concat([early_feature['block_2'], trans_layer], axis=3)
            layer_1 = slim.conv2d(x_input, num_outputs, conv_kernel_3, scope='conv1')
            layer_2 = slim.conv2d(layer_1, num_outputs, conv_kernel_3, scope='conv2')
            early_feature[scope_name] = layer_2

        scope_name = 'block_9'
        num_outputs = 64
        with tf.variable_scope(scope_name, reuse=reuse):
            trans_layer = slim.conv2d_transpose(
                layer_2, num_outputs, pool_kernel, pool_stride, scope='conv_trans')
            x_input = tf.concat([early_feature['block_1'], trans_layer], axis=3)
            layer_1 = slim.conv2d(x_input, num_outputs, conv_kernel_3, scope='conv1')
            layer_2 = slim.conv2d(layer_1, num_outputs, conv_kernel_3, scope='conv2')
            early_feature[scope_name] = layer_2

        scope_name = 'pred'
        with tf.variable_scope(scope_name, reuse=reuse):
            layer_1 = slim.conv2d(layer_2, 1, conv_kernel_1, scope='conv1',
                                  activation_fn=None, normalizer_fn=None)

            early_feature[scope_name] = layer_1

            # pred = tf.argmax(tf.nn.softmax(logits=layer_1), axis=3)
            pred = tf.sigmoid(layer_1)

        return tf.squeeze(layer_1), tf.squeeze(pred)


def build_loss(x_in, y_in, H, phase):
    '''
    This function builds the loss and accuracy
    '''
    im_width = H['im_width']
    im_height = H['im_height']
    batch_size = H['batch_size']
    start_ind = H['start_ind']
    valid_size = H['valid_size']
    num_class = H['num_class']
    epsilon = H['epsilon']
    apply_class_balancing = H['apply_class_balancing']

    logits, pred = build_pred(x_in, H, phase)
    y_crop = tf.cast(tf.slice(y_in, begin=[0, start_ind, start_ind],
                              size=[-1, valid_size, valid_size]), tf.float32)
    logits_crop = tf.slice(logits,
                           begin=[0, start_ind, start_ind],
                           size=[-1, valid_size, valid_size])
    pred_crop = tf.cast(tf.slice(pred,
                                 begin=[0, start_ind, start_ind],
                                 size=[-1, valid_size, valid_size]), tf.float32)
    if apply_class_balancing:
        class_weight = data_utils.calculate_class_weights()\
            [data_utils.CLASSES[class_type + 1]]
    # formulation of weighted cross entropy loss, dice index: https://arxiv.org/pdf/1707.03237.pdf
    if H['loss_function'] == 'cross_entropy':
        if apply_class_balancing:
            loss = tf.reduce_mean(
                tf.nn.weighted_cross_entropy_with_logits(
                    targets=y_crop,
                    logits=logits_crop, pos_weight=1. / class_weight))
        else:
            loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=y_crop, logits=logits_crop))

    elif H['loss_function'] == 'dice':

        intersection = tf.reduce_sum(tf.multiply(y_crop, pred_crop))
        union = tf.reduce_sum(tf.square(y_crop)) + \
                tf.reduce_sum(tf.square(pred_crop))
        loss = 1. - 2 * intersection / (union + tf.constant(epsilon))

    elif H['loss_function'] == 'jaccard':
        intersection = tf.reduce_sum(tf.multiply(y_crop, pred_crop))
        union = tf.reduce_sum(y_crop) + tf.reduce_sum(pred_crop) - intersection
        loss = 1. - intersection / (union + tf.constant(epsilon))

    elif H['loss_function'] == 'combo-jaccard':
        if apply_class_balancing:
            cen_loss = tf.reduce_mean(
                tf.nn.weighted_cross_entropy_with_logits(
                    targets=y_crop,
                    logits=logits_crop, pos_weight=1. / class_weight))
        else:
            cen_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=y_crop, logits=logits_crop))

        intersection = tf.reduce_sum(tf.multiply(y_crop, pred_crop))
        union = tf.reduce_sum(y_crop) + tf.reduce_sum(pred_crop) - intersection
        jaccard_loss = - tf.log((intersection + tf.constant(epsilon)) / \
                                (union + tf.constant(epsilon)))
        loss = cen_loss + jaccard_loss

    elif H['loss_function'] == 'combo-dice':
        if apply_class_balancing:
            cen_loss = tf.reduce_mean(
                tf.nn.weighted_cross_entropy_with_logits(
                    targets=y_crop,
                    logits=logits_crop, pos_weight=1. / class_weight))
        else:
            cen_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=y_crop, logits=logits_crop))

        intersection = tf.reduce_sum(tf.multiply(y_crop, pred_crop))
        union = tf.reduce_sum(y_crop) + tf.reduce_sum(pred_crop) - intersection
        dice_loss = - tf.log((intersection + tf.constant(epsilon)) / \
                             (union + tf.constant(epsilon)))
        loss = cen_loss + dice_loss

    pred_thres = tf.cast(tf.greater(pred_crop, 0.5), tf.float32)
    inter = tf.reduce_sum(tf.multiply(tf.cast(y_crop, tf.float32), pred_thres))
    uni = tf.reduce_sum(tf.cast(y_crop, tf.float32)) + \
          tf.reduce_sum(pred_thres) - inter
    jaccard = inter / (uni + tf.constant(epsilon))

    return loss, jaccard, logits_crop, pred_crop


def build(queues, H):
    '''
    This function returns the train operation, summary, global step
    '''
    im_width = H['im_width']
    im_height = H['im_height']
    num_class = H['num_class']
    num_channel = H['num_channel']
    batch_size = H['batch_size']
    log_dir = H['log_dir']
    norm_threshold = H['norm_threshold']

    loss, accuracy, x_in, y_in, logits, pred = {}, {}, {}, {}, {}, {}
    for phase in ['train', 'validate']:
        x_in[phase], y_in[phase] = queues[phase].dequeue_many(batch_size)
        loss[phase], accuracy[phase], logits[phase], pred[phase] = \
            build_loss(x_in[phase], y_in[phase], H, phase)

    learning_rate = tf.placeholder(dtype=tf.float32)
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.8, beta2=0.99)
    global_step = tf.Variable(0, trainable=False)
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss['train'], tvars)
    grads, norm = tf.clip_by_global_norm(grads, norm_threshold)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.apply_gradients(zip(grads, tvars), global_step=global_step)

    for phase in ['train', 'validate']:
        tf.summary.scalar(name=phase + '/loss', tensor=loss[phase])
        tf.summary.scalar(name=phase + '/accuracy', tensor=accuracy[phase])

        mean, var = tf.nn.moments(logits[phase], axes=[0, 1, 2])
        tf.summary.scalar(name=phase + '/logits/mean', tensor=mean)
        tf.summary.scalar(name=phase + '/logits/variance', tensor=var)

        mean, var = tf.nn.moments(pred[phase], axes=[0, 1, 2])
        tf.summary.scalar(name=phase + '/pred/mean', tensor=mean)
        tf.summary.scalar(name=phase + '/pred/variance', tensor=var)

    summary_op = tf.summary.merge_all()
    return loss, accuracy, train_op, summary_op, learning_rate, global_step
