package tests.batches;

import utest.Assert;
import convnethx.Vol;

class VolTest {

    public function new() {

    }

    public function test_volume_creation():Void {
        var vol:Vol = new Vol(2, 2, 1);

        Assert.equals(2, vol.sx);
        Assert.equals(2, vol.sy);
        Assert.equals(1, vol.depth);
        Assert.equals(4, vol.w.length);
    }

    public function test_volume_random():Void {
        var vol:Vol = new Vol(4, 4, 5);

        for (value in vol.w) Assert.floatEquals(0, value, 1);
    }

    public function test_volume_with_zeroes():Void {
        var vol:Vol = new Vol(4, 4, 5, 0);

        for (value in vol.w) Assert.equals(0, value);
    }

    public function test_predefined_values():Void {
        var vol:Vol = new Vol([0.1, 0.2, 0.3, 0.4]);

        Assert.equals(1, vol.sx);
        Assert.equals(1, vol.sy);
        Assert.equals(4, vol.depth);

        Assert.floatEquals(0.1, vol.get(0, 0, 0));
        Assert.floatEquals(0.2, vol.get(0, 0, 1));
        Assert.floatEquals(0.3, vol.get(0, 0, 2));
        Assert.floatEquals(0.4, vol.get(0, 0, 3));
    }

    public function test_volume_weight_operations():Void {
        var vol:Vol = new Vol(2, 2, 2, 0);

        vol.set(0, 1, 1, 5);
        Assert.floatEquals(5, vol.get(0, 1, 1));

        vol.add(0, 1, 0, 2);
        Assert.floatEquals(2, vol.get(0, 1, 0));

        vol.add(0, 1, 0, 2);
        Assert.floatEquals(4, vol.get(0, 1, 0));
    }

    public function test_volume_grad_weight_operations():Void {
        var vol:Vol = new Vol(2, 2, 2, 0);

        vol.set_grad(0, 1, 1, 5);
        Assert.floatEquals(5, vol.get_grad(0, 1, 1));

        vol.add_grad(0, 1, 0, 2);
        Assert.floatEquals(2, vol.get_grad(0, 1, 0));

        vol.add_grad(0, 1, 0, 2);
        Assert.floatEquals(4, vol.get_grad(0, 1, 0));
    }
}
