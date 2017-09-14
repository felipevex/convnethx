package convnethx.layer.loss;

import convnethx.model.json.JsonLayerSoftmax;
import convnethx.type.LayerType;
import convnethx.model.LayerOptionValue;
import haxe.io.Float64Array;

class LayerSoftmax extends Layer {

    public var es:Float64Array;

    public function new(option:LayerOptionValue) {
        super();
        this.layer_type = LayerType.SOFTMAX;

        var in_sx:Int = option.in_sx == null ? 0 : option.in_sx;
        var in_sy:Int = option.in_sy == null ? 0 : option.in_sy;
        var in_depth:Int = option.in_depth == null ? 0 : option.in_depth;

        // computed
        this.num_inputs = in_sx * in_sy * in_depth;
        this.out_depth = this.num_inputs;
        this.out_sx = 1;
        this.out_sy = 1;
    }

    override public function forward(V:Vol, is_training:Bool = false):Vol {
        this.in_act = V;

        var A:Vol = new Vol(1, 1, this.out_depth, 0);

        // compute max activation
        var as:Float64Array = V.w;
        var amax:Float = V.w[0];

        for (i in 1 ... this.out_depth) if (as[i] > amax) amax = as[i];

        // compute exponentials (carefully to not blow up)
        var es:Float64Array = Utils.zeros(this.out_depth);
        var esum:Float = 0;

        for (i in 0 ... this.out_depth) {
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

    override public function backward(y:Null<Int> = null):Null<Float> {
        // compute and accumulate gradient wrt weights and bias of this layer
        var x:Vol = this.in_act;

        x.dw = Utils.zeros(x.w.length); // zero out the gradient of input Vol

        for (i in 0 ... this.out_depth) {
            var indicator:Float = i == y ? 1.0 : 0.0;
            var mul:Float = -(indicator - this.es[i]);

            x.dw[i] = mul;
        }

        // loss is the class negative log likelihood
        return -Math.log(this.es[y]);
    }

    public function toJSON():JsonLayerSoftmax {
        var json:JsonLayerSoftmax = {
            layer_type : this.layer_type,
            out_depth : this.out_depth,
            out_sx : this.out_sx,
            out_sy : this.out_sy,
            num_inputs : this.num_inputs
        };

        return json;
    }

    public function fromJSON(json:JsonLayerSoftmax) {
        this.out_depth = json.out_depth;
        this.out_sx = json.out_sx;
        this.out_sy = json.out_sy;
        this.num_inputs = json.num_inputs;
    }
}
