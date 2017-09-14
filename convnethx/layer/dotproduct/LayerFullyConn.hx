package convnethx.layer.dotproduct;

import convnethx.model.ParamsAndGradsValue;
import convnethx.model.json.JsonLayerFC;
import convnethx.type.LayerType;
import convnethx.model.LayerOptionValue;
import haxe.io.Float64Array;

class LayerFullyConn extends Layer {

    public function new(option:LayerOptionValue) {
        super();
        this.layer_type = LayerType.FC;

        // todo must allow filter words??
        // this.out_depth = opt.num_neurons != null ? opt.num_neurons : opt.filters;
        this.out_depth = option.num_neurons == null ? 1 : option.num_neurons;

        // optional
        this.l1_decay_mul = option.l1_decay_mul != null ? option.l1_decay_mul : 0.0;
        this.l2_decay_mul = option.l2_decay_mul != null ? option.l2_decay_mul : 1.0;

        var in_sx:Int = option.in_sx == null ? 0 : option.in_sx;
        var in_sy:Int = option.in_sy == null ? 0 : option.in_sy;
        var in_depth:Int = option.in_depth == null ? 0 : option.in_depth;

        // computed
        this.num_inputs = in_sx * in_sy * in_depth;
        this.out_sx = 1;
        this.out_sy = 1;

        // initializations
        this.filters = [for(i in 0 ... this.out_depth) new Vol(1, 1, this.num_inputs)];

        var bias:Float = option.bias_pref != null ? option.bias_pref : 0;
        this.biases = new Vol(1, 1, this.out_depth, bias);
    }

    override public function forward(V:Vol, is_training:Bool = false):Vol {
        // INFO:
        // out_depth == opt.num_neurons
        // num_inputs == opt.in_sx * opt.in_sy * opt.in_depth

        this.in_act = V;

        var A:Vol = new Vol(1, 1, this.out_depth, 0);
        var Vw:Float64Array = V.w;

        for (i in 0 ... this.out_depth) {
            var a:Float = 0.0;
            var wi:Float64Array = this.filters[i].w;

            for (j in 0 ... this.num_inputs) {
                a += Vw[j] * wi[j]; // for efficiency use Vols directly for now
            }

            a += this.biases.w[i];
            A.w[i] = a;
        }

        this.out_act = A;
        return this.out_act;
    }

    override public function backward(y:Null<Int> = null):Null<Float> {
        var V:Vol = this.in_act;

        V.dw = Utils.zeros(V.w.length); // zero out the gradient in input Vol

        // compute gradient wrt weights and data
        for (i in 0 ... this.out_depth) {
            var tfi:Vol = this.filters[i];
            var chain_grad:Float = this.out_act.dw[i];

            for (d in 0 ... this.num_inputs) {
                V.dw[d] += tfi.w[d] * chain_grad; // grad wrt input data
                tfi.dw[d] += V.w[d] * chain_grad; // grad wrt params
            }

            this.biases.dw[i] += chain_grad;
        }

        return null;
    }

    override public function getParamsAndGrads():Array<ParamsAndGradsValue> {
        var response:Array<ParamsAndGradsValue> = [];

        for (i in 0 ... this.out_depth) {
            response.push(
                {
                    params: this.filters[i].w,
                    grads: this.filters[i].dw,
                    l1_decay_mul: this.l1_decay_mul,
                    l2_decay_mul: this.l2_decay_mul
                }
            );
        }

        response.push(
            {
                params: this.biases.w,
                grads: this.biases.dw,
                l1_decay_mul: 0.0,
                l2_decay_mul: 0.0
            }
        );

        return response;
    }

    public function toJSON():JsonLayerFC {
        var json:JsonLayerFC = {
            layer_type : this.layer_type,

            out_depth : this.out_depth,
            out_sx : this.out_sx,
            out_sy : this.out_sy,

            num_inputs : this.num_inputs,

            l1_decay_mul : this.l1_decay_mul,
            l2_decay_mul : this.l2_decay_mul,

            filters : [for (filter in this.filters) filter.toJSON()],

            biases : this.biases.toJSON()
        };

        return json;
    }

    public function fromJSON(json:JsonLayerFC):Void {
        this.out_depth = json.out_depth;
        this.out_sx = json.out_sx;
        this.out_sy = json.out_sy;
        this.num_inputs = json.num_inputs;
        this.l1_decay_mul = json.l1_decay_mul != null ? json.l1_decay_mul : 1.0;
        this.l2_decay_mul = json.l2_decay_mul != null ? json.l2_decay_mul : 1.0;

        this.filters = [];

        for(i in 0 ... json.filters.length) {
            var volume = new Vol(0, 0, 0, 0);
            volume.fromJSON(json.filters[i]);

            this.filters.push(volume);
        }

        this.biases = new Vol(0, 0, 0, 0);
        this.biases.fromJSON(json.biases);
    }
}
