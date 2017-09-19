package tests.batches;

import utest.Assert;
import convnethx.helper.LayerOptionHelper;
import convnethx.Net;
import convnethx.trainer.Trainer;
import convnethx.type.LayerType;
import convnethx.Vol;

class DemoTest {

    public function new() {

    }

    public function test_XOR():Void {
        var net:Net = new Net();
        net.makeLayers(
            [
                LayerOptionHelper.createInput(1, 1, 2),
                LayerOptionHelper.createFC(3, LayerType.TANH),
                LayerOptionHelper.createSoftmax(2)
            ]
        );

        var trainer:Trainer = new Trainer(net);


        for (iter in 0 ... 2000) {
            var point = new Vol([1.0, 1.0]);
            trainer.train(point, 0);

            var point = new Vol([1.0, 0.0]);
            trainer.train(point, 1);

            var point = new Vol([0.0, 1.0]);
            trainer.train(point, 1);

            var point = new Vol([0.0, 0.0]);
            trainer.train(point, 0);
        }

        var prediction = net.forward(new Vol([1.0, 1.0]));
        Assert.floatEquals(0, prediction.w[1], 0.01);

        var prediction = net.forward(new Vol([1.0, 0.0]));
        Assert.floatEquals(1, prediction.w[1], 0.01);

        var prediction = net.forward(new Vol([0.0, 1.0]));
        Assert.floatEquals(1, prediction.w[1], 0.01);

        var prediction = net.forward(new Vol([0.0, 0.0]));
        Assert.floatEquals(0, prediction.w[1], 0.01);

    }

}
