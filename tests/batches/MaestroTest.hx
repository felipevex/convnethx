package tests.batches;

import utest.Assert;
import convnethx.maestro.Maestro;

class MaestroTest {

    public function new() {

    }

    public function test_Direction():Void {
        //trace("\n\nTEST DIRECTION");

        var maestro:Maestro = new Maestro(2);
        maestro.addRule([0, 0, 0, 0], "CENTER");
        maestro.addRule([1, 0, 0, 0], "TOP");
        maestro.addRule([1, 0, 0, 1], "TOP-LEFT");
        maestro.addRule([1, 1, 0, 0], "TOP-RIGHT");
        maestro.addRule([0, 1, 0, 0], "RIGHT");
        maestro.addRule([0, 0, 1, 0], "BOTTOM");
        maestro.addRule([0, 0, 1, 1], "BOTTOM-LEFT");
        maestro.addRule([0, 1, 1, 0], "BOTTOM-RIGHT");
        maestro.addRule([0, 0, 0, 1], "LEFT");
        maestro.train(3000);

        Assert.equals("LEFT", maestro.test([0.1, 0.3, 0.1, 0.8]));
        Assert.equals("TOP-RIGHT", maestro.test([2.1, 1.3, 0.1, 0.8]));
        Assert.equals("BOTTOM-RIGHT", maestro.test([0.1, 0.6, 0.78, 0.2]));
    }

    public function test_XOR():Void {
        //trace("\n\nTEST XOR");

        var maestro:Maestro = new Maestro(12);
        maestro.addRule([1, 1], "FALSE");
        maestro.addRule([1, 0], "TRUE");
        maestro.addRule([0, 1], "TRUE");
        maestro.addRule([0, 0], "FALSE");
        maestro.train(100);

        Assert.equals("FALSE", maestro.test([1, 1]));
        Assert.equals("TRUE", maestro.test([1, 0]));
        Assert.equals("TRUE", maestro.test([0, 1]));
        Assert.equals("FALSE", maestro.test([0, 0]));
    }
}
