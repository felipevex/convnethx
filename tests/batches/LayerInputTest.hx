package tests.batches;

import convnethx.helper.LayerOptionHelper;
import convnethx.LayerInput;
import utest.Assert;

class LayerInputTest {

    public function new() {

    }

    public function test_layer_creation():Void {
        var layer:LayerInput = new LayerInput(LayerOptionHelper.createInput(1, 1, 2));

        Assert.isFalse(layer == null);
    }
}
