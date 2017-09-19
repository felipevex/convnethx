package tests;

import tests.batches.MaestroTest;
import tests.batches.DemoTest;
import tests.batches.NetTest;
import tests.batches.LayerInputTest;
import tests.batches.VolTest;
import tests.batches.UtilsTest;
import utest.ui.Report;
import utest.Runner;

class Main {

    public static function main() {

        var runner:Runner = new Runner();

        runner.addCase(new UtilsTest());
        runner.addCase(new VolTest());
        runner.addCase(new LayerInputTest());
        runner.addCase(new NetTest());
        runner.addCase(new DemoTest());
        runner.addCase(new MaestroTest());

        Report.create(runner);

        runner.run();
    }
}
