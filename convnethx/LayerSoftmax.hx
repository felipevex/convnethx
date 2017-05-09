package convnethx;

import haxe.io.Float64Array;

class LayerSoftmax extends Layer {

    public var es:Float64Array;

    public function new(opt:Opt) {
        super(opt);

        // computed
        this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
        this.out_depth = this.num_inputs;
        this.out_sx = 1;
        this.out_sy = 1;

        this.layer_type = LayerType.SOFTMAX;
    }

    override public function forward(V:Vol, is_training:Bool = false):Vol {
        this.in_act = V;

        var A:Vol = new Vol(1, 1, this.out_depth, [0]);

        // compute max activation
        var as:Float64Array = V.w;
        var amax:Float = V.w[0];

        for(i in 1 ... this.out_depth) {
            if (as[i] > amax) amax = as[i];
        }

        // compute exponentials (carefully to not blow up)
        var es:Float64Array = Utils.zeros(this.out_depth);
        var esum:Float = 0.0;

        for(i in 0 ... this.out_depth) {
            var e:Float = Math.exp(as[i] - amax);
            esum += e;
            es[i] = e;
        }

        // normalize and output to sum to one
        for (i in 0 ... this.out_depth) {
            es[i] /= esum;
            A.w[i] = es[i];
        }

        this.es = es; // save these for backprop
        this.out_act = A;

        return this.out_act;
    }

    override public function backward(y:Array<Float>):Float {
        // compute and accumulate gradient wrt weights and bias of this layer
        var x:Vol = this.in_act;
        x.dw = Utils.zeros(x.w.length); // zero out the gradient of input Vol

        var yValue:Float = y[0];

        for(i in 0 ... this.out_depth) {
            var indicator:Float = i == yValue ? 1.0 : 0.0;
            var mul:Float = -(indicator - this.es[i]);

            x.dw[i] = mul;
        }

        // loss is the class negative log likelihood
        return - Math.log(this.es[y]);
    }

    override public function toJSON():Dynamic {
        var json:Dynamic = {};

        json.out_depth = this.out_depth;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.layer_type = this.layer_type;
        json.num_inputs = this.num_inputs;

        return json;
    }

    override public function fromJSON(json:Dynamic) {
        this.out_depth = json.out_depth;
        this.out_sx = json.out_sx;
        this.out_sy = json.out_sy;
        this.layer_type = json.layer_type;
        this.num_inputs = json.num_inputs;
    }
}
