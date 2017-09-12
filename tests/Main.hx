package tests;

import tests.batches.NetTest;
import tests.batches.LayerInputTest;
import tests.batches.VolTest;
import tests.batches.UtilsTest;
import utest.ui.Report;
import utest.Runner;

class Main {

    static function main() {
        var runner:Runner = new Runner();

        runner.addCase(new UtilsTest());
        runner.addCase(new VolTest());
        runner.addCase(new LayerInputTest());
        runner.addCase(new NetTest());

        Report.create(runner);

        runner.run();
    }
}
