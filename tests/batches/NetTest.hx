package tests.batches;

import convnethx.LayerFullyConn;
import convnethx.type.LayerType;
import convnethx.LayerInput;
import convnethx.helper.NetHelper;
import convnethx.Layer;
import utest.Assert;
import convnethx.helper.LayerOptionHelper;
import convnethx.layer.model.LayerOption;

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
        var layerOptions:Array<LayerOption> = [
            LayerOptionHelper.createInput(1, 1, 2),
            LayerOptionHelper.createFC(5, LayerType.TANH),
            LayerOptionHelper.createFC(5, LayerType.TANH),
            LayerOptionHelper.createSoftMax(3)
        ];

        var layers:Array<Layer> = NetHelper.createLayers(layerOptions);
        
        Assert.equals(7, layerOptions.length);
        Assert.is(layers[0], LayerInput);
        Assert.is(layers[1], LayerFullyConn);
        Assert.is(layers[2], LayerFullyConn);
    }
}
