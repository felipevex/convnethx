package tests.batches;

import convnethx.model.DefMaxMinValue;
import utest.Assert;
import convnethx.Utils;

class UtilsTest {

    public function new() {

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
