package tests.batches;

import convnethx.layer.loss.LayerSoftmax;
import convnethx.layer.nonlinearities.LayerTanh;
import convnethx.layer.input.LayerInput;
import convnethx.layer.dotproduct.LayerFullyConn;
import convnethx.type.LayerType;
import convnethx.helper.NetHelper;
import convnethx.Layer;
import convnethx.helper.LayerOptionHelper;
import convnethx.model.LayerOptionValue;
import utest.Assert;

class NetTest {

    public function new() {

    }

    public function setup():Void {

//        this.net = new Net();
//
//        var layerDefs:Array<LayerOptionBase> [
//            LayerOptionHelper.createInput(1, 1, 2)
//        ];
//
//        this.net.makeLayers(layerDefs);

    }

    public function teardown():Void {
//        this.net = null;
    }

    public function test_net_layer_creation():Void {
        var layerOptions:Array<LayerOptionValue> = [
            LayerOptionHelper.createInput(1, 1, 2),
            LayerOptionHelper.createFC(5, LayerType.TANH),
            LayerOptionHelper.createFC(5, LayerType.TANH),
            LayerOptionHelper.createSoftmax(3)
        ];

        var layers:Array<Layer> = NetHelper.createLayers(layerOptions);

        Assert.equals(7, layers.length);
        Assert.is(layers[0], LayerInput);
        Assert.is(layers[1], LayerFullyConn);
        Assert.is(layers[2], LayerTanh);
        Assert.is(layers[3], LayerFullyConn);
        Assert.is(layers[4], LayerTanh);
        Assert.is(layers[5], LayerFullyConn);
        Assert.is(layers[6], LayerSoftmax);
    }
}
