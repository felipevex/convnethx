package convnethx;

import haxe.io.Float64Array;

class LayerFullyConn extends Layer {

    public function new(opt:Opt) {
        super(opt);

        // required
        // ok fine we will allow 'filters' as the word as well
        this.out_depth = opt.num_neurons != null ? opt.num_neurons : opt.filters;

        // optional
        this.l1_decay_mul = opt.l1_decay_mul != null ? opt.l1_decay_mul : 0.0;
        this.l2_decay_mul = opt.l2_decay_mul != null ? opt.l2_decay_mul : 1.0;

        // computed
        this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
        this.out_sx = 1;
        this.out_sy = 1;
        this.layer_type = LayerType.FC;

        // initializations
        var bias:Float = opt.bias_pref != null ? opt.bias_pref : 0.0;
        this.filters = [];

        for(i in 0 ... this.out_depth) {
            this.filters.push(
                new Vol(1, 1, this.num_inputs)
            );
        }

        this.biases = new Vol(1, 1, this.out_depth, [bias]);
    }

    override public function forward(V:Vol, is_training:Bool):Vol {
        this.in_act = V;

        var A:Vol = new Vol(1, 1, this.out_depth, [0.0]);
        var Vw:Float64Array = V.w;

        for (i in 0 ... this.out_depth) {
            var a:Float = 0.0;
            var wi:Float64Array = this.filters[i].w;

            for(d in 0 ... this.num_inputs) {
                a += Vw[d] * wi[d]; // for efficiency use Vols directly for now
            }

            a += this.biases.w[i];
            A.w[i] = a;
        }

        this.out_act = A;
        return this.out_act;
    }

    override public function backward(y:Array<Float> = null):Float {
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

        return 0;
    }

    override public function getParamsAndGrads():Array<Dynamic> {
        var response:Array<Dynamic> = [];

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

    override public function toJSON():Dynamic {
        var json:Dynamic = {};

        json.out_depth = this.out_depth;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.layer_type = this.layer_type;
        json.num_inputs = this.num_inputs;
        json.l1_decay_mul = this.l1_decay_mul;
        json.l2_decay_mul = this.l2_decay_mul;

        json.filters = [];

        for(i in 0 ... this.filters.length) {
           json.filters.push(this.filters[i].toJSON());
        }

        json.biases = this.biases.toJSON();

        return json;
    }

    override public function fromJSON(json:Dynamic):Void {
        this.out_depth = json.out_depth;
        this.out_sx = json.out_sx;
        this.out_sy = json.out_sy;
        this.layer_type = json.layer_type;
        this.num_inputs = json.num_inputs;
        this.l1_decay_mul = json.l1_decay_mul != null ? json.l1_decay_mul : 1.0;
        this.l2_decay_mul = json.l2_decay_mul != null ? json.l2_decay_mul : 1.0;

        this.filters = [];

        for(i in 0 ... json.filters.length) {
            var v = new Vol(0, 0, 0, [0]);
            v.fromJSON(json.filters[i]);

            this.filters.push(v);
        }

        this.biases = new Vol(0,0,0,[0]);
        this.biases.fromJSON(json.biases);
    }
}
