package tests.batches;

import haxe.io.Float64Array;
import convnethx.Vol;
import convnethx.trainer.Trainer;
import convnethx.Net;
import convnethx.layer.loss.LayerSoftmax;
import convnethx.layer.nonlinearities.LayerTanh;
import convnethx.layer.input.LayerInput;
import convnethx.layer.dotproduct.LayerFullyConn;
import convnethx.type.LayerType;
import convnethx.helper.LayerOptionHelper;
import convnethx.model.LayerOptionValue;
import utest.Assert;

class NetTest {

    var net:Net;
    var trainer:Trainer;

    public function new() {

    }

    public function setup():Void {

        this.net = new Net();

        var layerOptions:Array<LayerOptionValue> = [
            LayerOptionHelper.createInput(1, 1, 2),
            LayerOptionHelper.createFC(5, LayerType.TANH),
            LayerOptionHelper.createFC(5, LayerType.TANH),
            LayerOptionHelper.createSoftmax(3)
        ];

        this.net.makeLayers(layerOptions);

        this.trainer = new Trainer(
            this.net,
            {
                learning_rate:0.0001,
                momentum:0.0,
                batch_size:1,
                l2_decay:0.0
            }
        );

    }

    public function teardown():Void {
        this.net = null;
        this.trainer = null;
    }

    public function test_forward_volumes():Void {
        var x = new Vol([0.2, -0.3]);

        var probabilityVolume:Vol = this.net.forward(x);

        Assert.equals(3, probabilityVolume.w.length);

        var w:Float64Array = probabilityVolume.w;
        var sum:Float = 0;

        for (i in 0 ... 3) {
            Assert.isTrue(w.get(i) > 0);
            Assert.isTrue(w.get(i) < 1);

            sum += w.get(i);
        }

        Assert.floatEquals(1, sum);
    }

    // should increase probabilities for ground truth class when trained
    public function test_training():Void {
        // lets test 100 random point and label settings
        // note that this should work since l2 and l1 regularization are off
        // an issue is that if step size is too high, this could technically fail...

        for (i in 0 ...  100) {
            var x:Vol = new Vol(
                [
                    Math.random() * 2 - 1,
                    Math.random() * 2 - 1
                ]
            );

            var pv:Vol = net.forward(x);
            var gti:Int = Math.floor(Math.random() * 3);

            this.trainer.train(x, gti);

            var pv2:Vol = net.forward(x);

            Assert.isTrue(pv2.w.get(gti) > pv.w.get(gti));
        }
    }

    // should compute correct gradient at data
    public function test_gradient_compute():Void {
        // here we only test the gradient at data, but if this is
        // right then that's comforting, because it is a function
        // of all gradients above, for all layers.

        var x:Vol = new Vol(
            [
                Math.random() * 2 - 1,
                Math.random() * 2 - 1
            ]
        );

        var gti:Int = Math.floor(Math.random() * 3); // ground truth index

        this.trainer.train(x, gti); // computes gradients at all layers, and at x

        var delta:Float = 0.0001;

        for (i in 0 ... x.w.length) {
            var grad_analytic:Float = x.dw[i];

            var xold:Float = x.w[i];
            x.w[i] += delta;

            var c0:Float = net.getCostLoss(x, gti);

            x.w[i] -= 2 * delta;

            var c1:Float = net.getCostLoss(x, gti);

            x.w[i] = xold; // reset

            var grad_numeric:Float = (c0 - c1)/(2 * delta);
            var rel_error:Float = Math.abs(grad_analytic - grad_numeric) / Math.abs(grad_analytic + grad_numeric);

            trace(i + ': numeric: ' + grad_numeric + ', analytic: ' + grad_analytic + ' => rel error ' + rel_error);

            Assert.isTrue(rel_error < 1e-2);
        }
    }

    public function test_net_layer_creation():Void {
        Assert.equals(7, this.net.layers.length);
        Assert.is(this.net.layers[0], LayerInput);
        Assert.is(this.net.layers[1], LayerFullyConn);
        Assert.is(this.net.layers[2], LayerTanh);
        Assert.is(this.net.layers[3], LayerFullyConn);
        Assert.is(this.net.layers[4], LayerTanh);
        Assert.is(this.net.layers[5], LayerFullyConn);
        Assert.is(this.net.layers[6], LayerSoftmax);
    }
}
