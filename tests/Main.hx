package tests;

import tests.batches.UtilsTest;
import utest.ui.Report;
import utest.Runner;

class Main {

    static function main() {
        var runner:Runner = new Runner();

        runner.addCase(new UtilsTest());

        Report.create(runner);

        runner.run();


    }
}
