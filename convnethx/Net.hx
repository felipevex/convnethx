package convnethx;

/**
* Net manages a set of layers For now
* constraints: Simple linear order
* of layers, first layer input last layer a cost layer
**/
import convnethx.layer.dotproduct.LayerFullyConn;
import convnethx.layer.loss.LayerSoftmax;
import convnethx.layer.nonlinearities.LayerTanh;
import convnethx.layer.input.LayerInput;
import convnethx.helper.NetHelper;
import convnethx.type.LayerType;
import convnethx.Utils;
import haxe.io.Float64Array;

class Net {

    public var layers:Array<Layer>;

    public function new() {
        this.layers = [];
    }

    public function makeLayers(defs:Array<Opt>):Void {
        // few checks
        Utils.assert(defs.length >= 2, 'Error! At least one input layer and one loss layer are required.');
        Utils.assert(defs[0].type == LayerType.INPUT, 'Error! First layer must be the input layer, to declare size of inputs');

        // create the layers
        this.layers = NetHelper.createLayers(defs);
    }

    // forward prop the network.
    // The trainer class passes is_training = true, but when this function is
    // called from outside (not from the trainer), it defaults to prediction mode
    public function forward(V:Vol, is_training:Bool = false):Vol {

        var act:Vol = this.layers[0].forward(V, is_training);

        for (i in 1 ... this.layers.length) {
            act = this.layers[i].forward(act, is_training);
        }

        return act;
    }

    public function getCostLoss(V:Vol, y:Array<Float> = null):Null<Float> {
        this.forward(V, false);

        var N:Int = this.layers.length;
        var loss:Null<Float> = this.layers[N-1].backward(y);

        return loss;
    }

    public function backward(y:Array<Float>):Null<Float> {
        var N:Int = this.layers.length;

        var loss:Null<Float> = this.layers[N-1].backward(y); // last layer assumed to be loss layer

        var i:Int = N-2;

        while (i >= 0) {
            this.layers[i].backward();

            i--;
        }

        return loss;
    }

    public function getParamsAndGrads():Array<Dynamic> {
        // accumulate parameters and gradients for the entire network
        var response:Array<Dynamic> = [];

        for (i in 0 ... this.layers.length) {
            var layer_reponse:Array<Dynamic> = this.layers[i].getParamsAndGrads();

            for (j in 0 ... layer_reponse.length) {
                response.push(layer_reponse[j]);
            }
        }

        return response;
    }

    public function getPrediction():Int {
        // this is a convenience function for returning the argmax
        // prediction, assuming the last layer of the net is a softmax
        var S:Layer = this.layers[this.layers.length - 1];

        Utils.assert(S.layer_type == 'softmax', 'getPrediction function assumes softmax as last layer of the net!');

        var p:Float64Array = S.out_act.w;
        var maxv:Float = p[0];
        var maxi:Int = 0;

        for (i in 0 ... p.length) {
            if (p[i] > maxv) {
                maxv = p[i];
                maxi = i;
            }
        }

        return maxi; // return index of the class with highest class probability
    }

    public function toJSON():Array<Dynamic> {
        var json:Dynamic = {
            layers : []
        };

        for(i in 0 ... this.layers.length) {
            json.layers.push(this.layers[i].toJSON());
        }

        return json;
    }

    public function fromJSON(json:Dynamic):Void {
        this.layers = [];

        for(i in 0 ... json.layers.length) {
            var Lj:Dynamic = json.layers[i];
            var t:String = Lj.layer_type;
            var L:Layer;

            if(t == LayerType.INPUT) L = new LayerInput({});
            else if(t == LayerType.RELU) L = new LayerRelu({});
            else if(t == LayerType.SIGMOID) L = new LayerSigmoid({});
            else if(t == LayerType.TANH) L = new LayerTanh({});
            else if(t == LayerType.DROPOUT) L = new LayerDropout({});
            else if(t == LayerType.CONV) L = new LayerConv({});
            else if(t == LayerType.POOL) L = new LayerPool({});
            else if(t == LayerType.LRN) L = new LayerLocalResponseNormalization({});
            else if(t == LayerType.SOFTMAX) L = new LayerSoftmax({});
            else if(t == LayerType.REGRESSION) L = new LayerRegression({});
            else if(t == LayerType.FC) L = new LayerFullyConn({});
            else if(t == LayerType.MAXOUT) L = new LayerMaxout({});
            else if(t == LayerType.SVM) L = new LayerSVM({});
            else L = new Layer({});

            L.fromJSON(Lj);

            this.layers.push(L);
        }
    }
}
