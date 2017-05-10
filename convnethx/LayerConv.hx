package convnethx;

class LayerConv extends Layer {

    public function new(opt:Opt) {
        super(opt);

        this.out_depth = opt.filters;
        this.sx = opt.sx; // filter size. Should be odd if possible, it's cleaner.
        this.in_depth = opt.in_depth;
        this.in_sx = opt.in_sx;
        this.in_sy = opt.in_sy;

        // optional
        this.sy = opt.sy != null ? opt.sy : this.sx;
        this.stride = opt.stride != null ? opt.stride : 1; // stride at which we apply filters to input volume
        this.pad = opt.pad != null ? opt.pad : 0; // amount of 0 padding to add around borders of input volume
        this.l1_decay_mul = opt.l1_decay_mul != null ? opt.l1_decay_mul : 0.0;
        this.l2_decay_mul = opt.l2_decay_mul != null ? opt.l2_decay_mul : 1.0;

        // computed
        // note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
        // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
        // final application.
        this.out_sx = Math.floor((this.in_sx + this.pad * 2 - this.sx) / this.stride + 1);
        this.out_sy = Math.floor((this.in_sy + this.pad * 2 - this.sy) / this.stride + 1);
        this.layer_type = LayerType.CONV;

        // initializations
        var bias:Float = opt.bias_pref != null ? opt.bias_pref : 0.0;

        this.filters = [];

        for(i in 0 ... this.out_depth) {
            this.filters.push(
                new Vol(this.sx, this.sy, this.in_depth)
            );
        }

        this.biases = new Vol(1, 1, this.out_depth, [bias]);
    }

    override public function forward(V:Vol, is_training:Bool):Vol {
        // optimized code by @mdda that achieves 2x speedup over previous version

        this.in_act = V;

        var A:Vol = new Vol(this.out_sx | 0, this.out_sy | 0, this.out_depth | 0, [0.0]);

        var V_sx:Int = V.sx | 0;
        var V_sy:Int = V.sy | 0;
        var xy_stride:Int = this.stride | 0;

        for (d in 0 ... this.out_depth) {

            var f:Vol = this.filters[d];
            var x:Int = - this.pad | 0;
            var y:Int = - this.pad | 0;

            for (ay in 0 ... this.out_sy) { // xy_stride

                y += xy_stride
                x = -this.pad | 0;

                for (ax in 0 ... this.out_sx) { // xy_stride

                    x += xy_stride

                    // convolve centered at this particular location
                    var a:Float = 0.0;

                    for (fy in 0 ... f.sy) {

                        var oy:Int = y + fy; // coordinates in the original input array coordinates

                        for (fx in 0 ... f.sx) {

                            var ox:Int = x + fx;

                            if (oy >= 0 && oy < V_sy && ox >= 0 && ox < V_sx) {
                                for (fd in 0 ... f.depth) {
                                    // avoid function call overhead (x2) for efficiency, compromise modularity :(

                                    a += f.w[((f.sx * fy) + fx) * f.depth + fd] * V.w[((V_sx * oy)+ox) * V.depth + fd];
                                }
                            }
                        }
                    }

                    a += this.biases.w[d];

                    A.set(ax, ay, d, a);
                }
            }
        }

        this.out_act = A;

        return this.out_act;
    }

    override public function backward(y:Array<Float> = null):Float {

        var V:Vol = this.in_act;
        V.dw = Utils.zeros(V.w.length); // zero out gradient wrt bottom data, we're about to fill it

        var V_sx:Int = V.sx | 0;
        var V_sy:Int = V.sy | 0;
        var xy_stride:Int = this.stride | 0;

        for(d in 0 ... this.out_depth) {

            var f:Vol = this.filters[d];
            var x:Int = - this.pad | 0;
            var y:Int = -this.pad |0;

            for (ay in 0 ... this.out_sy) {// }}var ay=0; ay<this.out_sy; y+=xy_stride,ay++) {  // xy_stride
                y += xy_stride;
                x = -this.pad |0;

                for (ax in 0 ... this.out_sx) {  // xy_stride

                    x += xy_stride

                    // convolve centered at this particular location
                    var chain_grad:Float = this.out_act.get_grad(ax,ay,d); // gradient from above, from chain rule

                    for (fy in 0 ... f.sy) {
                        var oy:Int = y + fy; // coordinates in the original input array coordinates

                        for (fx in 0 ... f.sx) {

                            var ox:Int = x + fx;

                            if (oy >= 0 && oy < V_sy && ox >= 0 && ox < V_sx) {
                                for (fd in 0 ... f.depth) {
                                    // avoid function call overhead (x2) for efficiency, compromise modularity :(

                                    var ix1:Int = ((V_sx * oy) + ox) * V.depth + fd;
                                    var ix2:Int = ((f.sx * fy) + fx) * f.depth + fd;

                                    f.dw[ix2] += V.w[ix1] * chain_grad;
                                    V.dw[ix1] += f.w[ix2] * chain_grad;
                                }
                            }
                        }
                    }

                    this.biases.dw[d] += chain_grad;
                }
            }
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
                    l2_decay_mul: this.l2_decay_mul,
                    l1_decay_mul: this.l1_decay_mul
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

        json.sx = this.sx; // filter size in x, y dims
        json.sy = this.sy;
        json.stride = this.stride;
        json.in_depth = this.in_depth;
        json.out_depth = this.out_depth;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.layer_type = this.layer_type;
        json.l1_decay_mul = this.l1_decay_mul;
        json.l2_decay_mul = this.l2_decay_mul;
        json.pad = this.pad;

        json.filters = [];

        for(i in 0 ... this.filters.length) {
            json.filters.push(
                this.filters[i].toJSON()
            );
        }

        json.biases = this.biases.toJSON();

        return json;
    }

    override public function fromJSON(json:Dynamic):Void {

        this.out_depth = json.out_depth;
        this.out_sx = json.out_sx;
        this.out_sy = json.out_sy;
        this.layer_type = json.layer_type;
        this.sx = json.sx; // filter size in x, y dims
        this.sy = json.sy;
        this.stride = json.stride;
        this.in_depth = json.in_depth; // depth of input volume

        this.filters = [];

        this.l1_decay_mul = json.l1_decay_mul != null ? json.l1_decay_mul : 1.0;
        this.l2_decay_mul = json.l2_decay_mul != null ? json.l2_decay_mul : 1.0;
        this.pad = json.pad != null ? json.pad : 0;

        for(i in 0 ... json.filters.length) {
            var v:Vol = new Vol(0, 0, 0, [0]);
            v.fromJSON(json.filters[i]);
            this.filters.push(v);
        }

        this.biases = new Vol(0, 0, 0, [0]);
        this.biases.fromJSON(json.biases);

    }
}
