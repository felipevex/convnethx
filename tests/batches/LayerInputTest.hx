package tests.batches;

import convnethx.layer.input.LayerInput;
import convnethx.helper.LayerOptionHelper;
import utest.Assert;

class LayerInputTest {

    public function new() {

    }

    public function test_layer_creation():Void {
        var layer:LayerInput = new LayerInput(LayerOptionHelper.createInput(1, 1, 2));

        Assert.isFalse(layer == null);
    }
}
