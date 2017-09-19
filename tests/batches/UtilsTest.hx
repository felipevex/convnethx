package tests.batches;

import convnethx.model.DefMaxMinValue;
import utest.Assert;
import convnethx.Utils;

class UtilsTest {

    public function new() {

    }

    public function test_tahn():Void {
        Assert.floatEquals(0.761594156, Utils.tanh(1), 1e-9);
        Assert.floatEquals(0.96402758, Utils.tanh(2), 1e-9);
        Assert.floatEquals(0.995054754, Utils.tanh(3), 1e-9);
        Assert.floatEquals(0.022995945, Utils.tanh(0.023), 1e-9);
        Assert.floatEquals(0.022995945, Utils.tanh(0.023), 1e-9);
        Assert.floatEquals(0.001999997, Utils.tanh(0.002), 1e-9);
    }

    public function test_randf():Void {
        var rlist:Array<Float> = [for (i in 0 ... 50) Utils.randf(2, 4)];
        for (value in rlist) Assert.isTrue(value >= 2 && value < 4);
    }

    public function test_randi():Void {
        var rlist:Array<Float> = [for (i in 0 ... 50) Utils.randi(2, 4)];
        for (value in rlist) Assert.isTrue(value == 2 || value == 3);
    }

    public function test_randp():Void {
        var rlist:Array<Array<Int>> = [for (i in 0 ... 30) Utils.randperm(10)];

        for (value in rlist) {
            for (i in 0 ... 10) value.remove(i);
            Assert.isTrue(value.length == 0);
        }
    }

    public function test_maxmin():Void {
        var value:DefMaxMinValue = Utils.maxmin([-0.1, -1.2, 30, 5, 0.0001, 229.34]);

        Assert.equals(229.34, value.maxv);
        Assert.equals(5, value.maxi);
        Assert.equals(-1.2, value.minv);
        Assert.equals(1, value.mini);
        Assert.equals(230.54, value.dv);
    }

    public function test_arrcontain():Void {
        var data:Array<Dynamic> = [10, 29, "string", Date.now(), 9.4];

        Assert.isTrue(Utils.arrContains(data, 29));
        Assert.isTrue(Utils.arrContains(data, "string"));
    }
}
