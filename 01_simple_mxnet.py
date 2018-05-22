#
# https://github.com/dmlc/mxnet-notebooks/blob/master/python/moved-from-mxnet/class_active_maps.ipynb
# class active maps on mnist tutorial
# while the tutorial is using a pre-trained model
# we are going to build everthing from scratch here

from time import time
import mxnet as mx
from mxnet import gluon, autograd, metric
from mxboard import SummaryWriter
from mxnet.gluon.data.vision import transforms
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


class SimpleMNISTConv(gluon.HybridBlock):
    def __init__(self, no_class, ctx, **kw_args):
        super(SimpleMNISTConv, self).__init__(**kw_args)

        with self.name_scope():
            self.block = mx.gluon.nn.HybridSequential()
            self.block.add(mx.gluon.nn.Conv2D(channels = 16, kernel_size=(3, 3), padding=(0,0), activation='relu'))
            self.block.add(mx.gluon.nn.Conv2D(channels = 64, kernel_size=(1, 1), padding=(0,0)))
            self.block.add(mx.gluon.nn.MaxPool2D(pool_size=(2,2)))
            self.block.add(mx.gluon.nn.Conv2D(channels= 128, kernel_size=(1, 1), padding=(0,0)))
            self.block.add(mx.gluon.nn.MaxPool2D(pool_size=(2, 2), strides=2))
            #
            self.block.add(mx.gluon.nn.Flatten())

            self.block.add(mx.gluon.nn.Dense(units=128, activation='relu'))
            self.block.add(mx.gluon.nn.Dense(units=64, activation='relu'))

            self.block.add(mx.gluon.nn.Dense(no_class))

    def forward(self, x, *args):
        return self.block(x)


def acc(output, label):
    # output: (batch, num_output) float32 ndarray
    # label: (batch, ) int32 ndarray
    return (output.argmax(axis=1) ==
            label.astype('float32')).mean().asscalar()

ctx = mx.gpu()
batch_size = 100
num_class = 10

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.13, 0.31)]
)

# mnist_train = mnist_train.transform_first(transformer)

# load the data
train_data = mx.gluon.data.DataLoader(
    mx.gluon.data.vision.MNIST(train=True).transform_first(transformer),
    batch_size=batch_size, shuffle=True)

valid_data = mx.gluon.data.DataLoader(
    mx.gluon.data.vision.MNIST(train=False).transform_first(transformer),
    batch_size=batch_size, shuffle=False)

mynet = SimpleMNISTConv(num_class, ctx)
mynet.collect_params().initialize(force_reinit=True, ctx=ctx)

# this should be used in the standard training loop
# mxboard_logacc = mx.contrib.tensorboard.LogMetricsCallback(logging_dir="./logs")

sw = SummaryWriter(logdir='./logs')

train_acc  = metric.Accuracy()
val_acc    = metric.Accuracy()

# the composite metric allows us to track multiple metrics at once
# eval_metrics = mx.metric.CompositeEvalMetric()
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(mynet.collect_params(), 'sgd', {'learning_rate': 0.1})

print("Starting Training")
for epoch in range(10):
    train_loss, f_train_acc, f_val_acc = .0, .0, .0
    tic = time()
    for batch_data in train_data:
        data = batch_data[0].as_in_context(ctx)
        label = batch_data[1].as_in_context(ctx)
        # forward + backward
        with autograd.record():
            output = mynet(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        # update parameters
        trainer.step(batch_size)
        # calculate training metrics
        train_loss += loss.mean().asscalar()
        f_train_acc += acc(output, label)
    #
    # # calculate validation accuracy
    for batch_data in valid_data:
        data = batch_data[0].as_in_context(ctx)
        label = batch_data[1].as_in_context(ctx)
        f_val_acc += acc(mynet(data), label)

    f_train_acc = f_train_acc / len(train_data)
    f_val_acc = f_val_acc / len(train_data)
    sw.add_scalar(tag="RMNIST_mx_05_training_accuracy", value=f_train_acc, global_step=epoch)
    sw.add_scalar(tag="MNIST_mx_05_validation_accuracy", value=f_val_acc, global_step=epoch)
    #
    print("Epoch %d: Loss: %.3f, Train acc %.3f, Test acc %.3f, Time %.1f sec" % (
        epoch, train_loss/len(train_data), f_train_acc, f_val_acc, time()-tic))

