from time import time
import mxnet as mx
from mxnet import gluon, metric, autograd
from mxboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np


def acc(output, label):
    # output: (batch, num_output) float32 ndarray
    # label: (batch, ) int32 ndarray
    return (output.argmax(axis=1) ==
            label.astype('float32')).mean().asscalar()


def get_cam(conv_feat_map, weight_fc):
    assert len(weight_fc.shape) == 2
    if len(conv_feat_map.shape) == 3:
        C, H, W = conv_feat_map.shape
        assert weight_fc.shape[1] == C
        detection_map = weight_fc.dot(conv_feat_map.reshape(C, H*W))
        detection_map = detection_map.reshape(-1, H, W)
    elif len(conv_feat_map.shape) == 4:
        N, C, H, W = conv_feat_map.shape
        assert weight_fc.shape[1] == C
        M = weight_fc.shape[0]
        detection_map = np.zeros((N, M, H, W))
        for i in xrange(N):
            tmp_detection_map = weight_fc.dot(conv_feat_map[i].reshape(C, H*W))
            detection_map[i, :, :, :] = tmp_detection_map.reshape(-1, H, W)
    return detection_map



class MNistHybrid(gluon.nn.HybridBlock):
    def __init__(self, no_class, **kwargs):
        super(MNistHybrid, self).__init__(**kwargs)
        self.number_of_classes = no_class
        with self.name_scope():
            self.conv1 = gluon.nn.Conv2D(channels=30, kernel_size=(3, 3), activation='relu', padding=(0, 0))
            self.conv2 = gluon.nn.Conv2D(channels=50, kernel_size=(3, 3), activation='relu', padding=(0, 0))
            self.conv3 = gluon.nn.Conv2D(channels=100, kernel_size=(3, 3), activation='relu', padding=(0, 0))
            self.pool3 = gluon.nn.MaxPool2D(pool_size=(2,2))

            self.flatten1 = gluon.nn.Flatten()
            self.dense1 = gluon.nn.Dense(units=128)
            self.dense2 = gluon.nn.Dense(units=64)
            self.dense3 = gluon.nn.Dense(units=self.number_of_classes)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten1(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)

mx.random.seed(1)

ctx = mx.gpu()
batch_size = 50
no_class   = 10
epochs     = 1
lr         = 0.1
do_train   = False
sw = SummaryWriter(logdir="./logs")

mnist_transformer = gluon.data.vision.transforms.Compose(
    [
        gluon.data.vision.transforms.ToTensor()
    ]
)

train_data = mx.gluon.data.DataLoader(dataset = gluon.data.vision.MNIST(train=True).transform_first(mnist_transformer),
                                      batch_size=batch_size, shuffle=True)
test_data  = mx.gluon.data.DataLoader(dataset = gluon.data.vision.MNIST(train=True).transform_first(mnist_transformer),
                                      batch_size=batch_size, shuffle=False)

mnist_net = MNistHybrid(no_class=no_class)

# if we set verbose - the printed numbers will be a lot finer
mnist_net.collect_params().initialize(init=mx.init.Xavier(), force_reinit=True, verbose=False, ctx=ctx)
trainer = mx.gluon.Trainer(params=mnist_net.collect_params(), optimizer="sgd", optimizer_params={"learning_rate": lr})
loss_fun = mx.gluon.loss.SoftmaxCrossEntropyLoss()
train_accuracy = metric.Accuracy()
test_accuracy  = metric.Accuracy()
ninv_train = 1/len(train_data)
ninv_test  = 1/len(test_data)

# reshaped as batch, no_channel, w, h
sample_1 = mx.gluon.data.vision.MNIST(train=True)[0][0].reshape((1,1,28,28))

plt.imshow(sample_1.asnumpy().reshape((28,28)), cmap='gray')
plt.show()

if do_train:
    for a_epoch in range(epochs):
        train_loss, f_train_acc, f_val_acc = .0, .0, .0
        tic = time()
        for a_batch in train_data:
            data_i = a_batch[0].as_in_context(ctx)
            label_i= a_batch[1].as_in_context(ctx)

            with autograd.record(train_mode=True):
                y_hat = mnist_net(data_i)
                loss  = loss_fun(y_hat, label_i)
            loss.backward()
            trainer.step(batch_size=batch_size)
            train_loss  += loss.mean().asscalar()
            f_train_acc += acc(y_hat, label_i)

        for batch_data in test_data:
            data = batch_data[0].as_in_context(ctx)
            label = batch_data[1].as_in_context(ctx)
            f_val_acc += acc(mnist_net(data), label)

        f_train_acc = f_train_acc * ninv_train
        f_val_acc = f_val_acc * ninv_test

        sw.add_scalar(tag="RMNIST-07_training_accuracy", value=f_train_acc, global_step=a_epoch)
        sw.add_scalar(tag="MNIST-07_validation_accuracy", value=f_val_acc, global_step=a_epoch)
        #
        print("Epoch %d: Loss: %.3f, Train acc %.3f, Test acc %.3f, Time %.1f sec" % (
            a_epoch, train_loss/len(train_data), f_train_acc, f_val_acc, time()-tic))

# requires either full layer specification or we pass one batch of data through the net
# so that deferred initialization can work it's magic

no_in_channel = 50
no_out_channel = 100
w, h = 3, 3
s = (no_in_channel*no_out_channel, w, h)

# outputs = mod.predict(blob)
# score = outputs[0][0]
# conv_fm = outputs[1][0]

# weight_fc =  mnist_net.conv3.collect_params().get('weight').data().asnumpy().reshape(s)