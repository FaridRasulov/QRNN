import tensorflow as tf


class SentimentModel:
    def __init__(self, embeddings, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE, beta=4e-6):
        self.batch_size = BATCH_SIZE
        self.seq_len = SEQ_LEN
        self.vocab_size = VOCAB_SIZE
        self.embeddings = embeddings

        self.inputs = tf.placeholder(tf.int32, [BATCH_SIZE, SEQ_LEN], name="inputs")
        self.masks = tf.placeholder(tf.float32, [BATCH_SIZE, SEQ_LEN], name="mask")
        self.labels = tf.placeholder(tf.int32, [BATCH_SIZE], name="labels")

        x, weights = self.forward()
        loss = self.inference(x)
        
        self.op = tf.train.RMSPropOptimizer(0.001).minimize(loss)
        self.epoch = tf.Variable(0, dtype=tf.int32, trainable=False, name='epoch')

    def inference(self, x):
        masks = self.masks
        labels = self.labels
        outputs = tf.reduce_mean(x * tf.expand_dims(masks, -1), 1)
        logits = tf.layers.dense(tf.squeeze(outputs), 2)
        pred = tf.argmax(tf.nn.softmax(logits), -1)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
   
        correct_prediction = tf.equal(tf.cast(pred, tf.int32), labels)
        self.accuracy = tf.reduce_sum(tf.cast(correct_prediction,tf.float32)) / self.batch_size
        
        return loss


class DenseQRNNModel(SentimentModel):
    def forward(self):
        inputs = self.inputs

        num_layers = 1
        input_size = 50
        num_convs = 10
        conv_size = 50
        x = tf.expand_dims(tf.nn.embedding_lookup(self.embeddings, inputs), -1)
        qrnn = DenseQRNNLayers(input_size,conv_size,num_convs,range(num_layers),num_layers, dropout=0.3)
        x = qrnn(x)#, train=self.train)
        weights = [l.W for l in qrnn.layers] + [l.b for l in qrnn.layers]
        return tf.squeeze(x), weights

class QRNNLayer:
    def __init__(self, input_size, conv_size, hidden_size, layer_id, pool='fo', zoneout=0.0, num_in_channels=1):
        
        self.input_size = input_size
        self.conv_size = conv_size if conv_size%2==0 else conv_size+1
        self.hidden_size = hidden_size
        self.layer_id = layer_id
        self.pool = pool
        self.zoneout = zoneout
        self.num_in_channels = num_in_channels
        init = tf.random_normal_initializer()
        filter_shape = [conv_size, input_size, num_in_channels, hidden_size*(len(pool)+1)]

        with tf.variable_scope('QRNN/conv/'+str(layer_id)):
            self.W = tf.get_variable('W', filter_shape, initializer=init, dtype=tf.float32)
            self.b = tf.get_variable('b', [hidden_size*(len(pool)+1)], initializer=init, dtype=tf.float32)

    def __call__(self, inputs):
        gates = self.conv(inputs)
        if self.zoneout and self.zoneout > 0.0:
            F = gates[2]
            F = 1-tf.nn.dropout(F, 1-self.zoneout)
            gates[2] = F
        if self.pool == 'f': return self.f_pool(gates)
        elif self.pool == 'fo': return self.fo_pool(gates)
        elif self.pool == 'ifo': return self.ifo_pool(gates)

    def conv(self, inputs):
        padded_inputs = tf.pad(inputs, [[0, 0], [self.conv_size - 1, 0], [0, 0], [0, 0]], "CONSTANT")
        conv = tf.nn.conv2d(padded_inputs, self.W, strides=[1, 1, 1, 1],padding='VALID', name='conv'+str(self.layer_id))
        conv += self.b
        gates = tf.split(conv, (len(self.pool)+1), 3)
        gates[0] = tf.tanh(gates[0])
        for i in range(1, len(gates)):
            gates[i] = tf.sigmoid(gates[i])
        return gates

    def unstack(self, gates, pooling):
        if pooling == 'f': Z, F = gates
        elif pooling == 'fo': Z, F, O = gates
        elif pooling == 'ifo': Z, F, O, I = gates

        Z = tf.unstack(Z, axis=1)
        F = tf.unstack(F, axis=1)
        if pooling == 'f': return Z,F
        O = tf.unstack(O, axis=1)
        if pooling == 'fo': return Z,F,O
        I = tf.unstack(O, axis=1)
        if pooling == 'ifo': return Z,F,O,I

    def f_pool(self, gates):
        Z, F = self.unstack(gates,'f')
        H = tf.zeros(tf.shape(Z[0]), tf.float32) #[tf.fill(tf.shape(Z[0]), 0.0)]
        for i in range(len(Z)):
            h = tf.multiply(F[i], H[-1]) + tf.multiply(1-F[i], Z[i])
            H.append(h)
        H = tf.stack(H[1:], axis=1)
        return tf.transpose(H, perm=[0, 1, 3, 2])

    def fo_pool(self, gates):
        Z, F, O = self.unstack(gates,'fo')
        C = [tf.fill(tf.shape(Z[0]), 0.0)]
        H = []
        for i in range(len(Z)):
            c = tf.multiply(F[i], C[-1]) + tf.multiply(1-F[i], Z[i])
            h = tf.multiply(O[i], c)
            C.append(c)
            H.append(h)
        H = tf.stack(H, axis=1)
        return tf.transpose(H, perm=[0, 1, 3, 2])       

    def ifo_pool(self, gates):
        Z, F, O, I = self.unstack(gates,'ifo')
        C = [tf.fill(tf.shape(Z[0]), 0.0)]
        H = []
        for i in range(len(Z)):
            c = tf.multiply(F[i], C[-1]) + tf.multiply(I[i], Z[i])
            h = tf.multiply(O[i], c)
            C.append(c)
            H.append(h)
        H = tf.stack(H, axis=1)
        return tf.transpose(H, perm=[0, 1, 3, 2])

class DenseQRNNLayers:
    def __init__(self, input_size, conv_size, hidden_size, layer_ids, num_layers, zoneout=0.0, dropout=0.0):
        self.layers = []
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.layers = [QRNNLayer(hidden_size, conv_size, hidden_size,layer_ids[i], pool='fo', zoneout=zoneout, num_in_channels=i+1) for i in range(num_layers)]

    def __call__(self, inputs):
        inputs = tf.layers.dense(tf.transpose(inputs, [0, 1, 3, 2]), self.hidden_size)
        inputs = tf.transpose(inputs, [0, 1, 3, 2])
        for layer in self.layers:
            outputs = layer(inputs)
            if self.dropout and self.dropout > 0:
                keep_prob = 1 - self.dropout
                outputs = tf.nn.dropout(outputs, keep_prob)
            inputs = tf.concat([inputs, outputs], 3)
        return tf.squeeze(outputs[:, :, :, -1])