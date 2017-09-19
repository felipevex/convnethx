package demo;

import priori.app.PriApp;

class Main extends PriApp {

    public function new() {
        super();
    }

    override private function setup():Void {

        tests.Main.main();


//        this.example1();
//        this.example2();
//        this.example3();
    }

    override private function paint():Void {

    }

    /*
    private function example1():Void {
        var v:Vol = new Vol(32, 32, 3);
        trace(v.toJSON());

        var v:Vol = new Vol(32, 32, 3, [0]); // same volume but init with zeros
        trace(v.toJSON());

        var v:Vol = new Vol(1, 1, 3); // a 1x1x3 Vol with random numbers
        trace(v.toJSON());

        var v:Vol = new Vol(1, 1, 3, [1.2, 3.5, 3.6]);
        trace(v.toJSON());

        // the Vol is a wrapper around two lists: .w and .dw, which both have
        // sx * sy * depth number of elements. E.g:
        trace(v.w[0]); // contains 1.2
        trace(v.dw[0]); // contains 0, because gradients are initialized with zeros

        // you can also access the 3-D Vols with getters and setters
        // but these are subject to function call overhead
        var vol3d:Vol = new Vol(10, 10, 5);
        vol3d.set(2, 0, 1, 5.0); // set coordinate (2,0,1) to 5.0
        trace(vol3d.get(2, 0, 1)); // returns 5.0
    }

    private function example2():Void {
        var layer_defs:Array<Opt> = [];

        // minimal network: a simple binary SVM classifer in 2-dimensional space
        layer_defs.push(
            {
                type : LayerType.INPUT,
                out_sx : 1,
                out_sy : 1,
                out_depth : 2
            }
        );

        layer_defs.push(
            {
                type : LayerType.SVM,
                num_classes : 2
            }
        );

        // create a net
        var net:Net = new Net();
        net.makeLayers(layer_defs);

        // create a 1x1x2 volume of input activations:
        var x:Vol = new Vol(1, 1, 2, [0.5, -1.3]);

        // a shortcut for the above is var x = new convnetjs.Vol([0.5, -1.3]);

        var scores:Vol = net.forward(x); // pass forward through network
        // scores is now a Vol() of output activations
        trace('score for class 0 is assigned:'  + scores.w[0]);

        trace(net.toJSON());
    }

    private function example3():Void {
        var layer_defs:Array<Opt> = [];
        // input layer of size 1x1x2 (all volumes are 3D)

        layer_defs.push(
            {
                type:LayerType.INPUT,
                out_sx:1,
                out_sy:1,
                out_depth:2
            }
        );

        // some fully connected layers
        layer_defs.push(
            {
                type:LayerType.FC,
                num_neurons : 20,
                activation:LayerType.RELU
            }
        );

        layer_defs.push(
            {
                type:LayerType.FC,
                num_neurons : 20,
                activation:LayerType.RELU
            }
        );

        // a softmax classifier predicting probabilities for two classes: 0,1
        layer_defs.push(
            {
                type:LayerType.SOFTMAX,
                num_classes : 2
            }
        );


        // create a net out of it
        var net:Net = new Net();
        net.makeLayers(layer_defs);

        // the network always works on Vol() elements. These are essentially
        // simple wrappers around lists, but also contain gradients and dimensions
        // line below will create a 1x1x2 volume and fill it with 0.5 and -1.3
        var x:Vol = new Vol(1, 1, 2, [0.5, -1.3]);

        var probability_volume:Vol = net.forward(x);
        trace('probability that x is class 0: ' + probability_volume.w[0]);
//        trace('probability that x is class 1: ' + probability_volume.w[1]);
//        trace(probability_volume.w[0] + probability_volume.w[1]);

        // prints 0.50101

        trace(net.toJSON());

//        var trainer:Trainer = new Trainer(
//            net,
//            {
//                learning_rate : 0.01,
//                l2_decay : 0.001
//            }
//        );
//
//        trainer.train(x, 0);
//
//        var probability_volume2 = net.forward(x);
//        console.log('probability that x is class 0: ' + probability_volume2.w[0]);
//        // prints 0.50374
    }
    */
}
