package convnethx;

import haxe.io.Float64Array;

class LayerPool extends Layer {

    public var switchx:Float64Array;
    public var switchy:Float64Array;

    public function new(opt:Opt) {
        super(opt);

        // required
        this.sx = opt.sx; // filter size
        this.in_depth = opt.in_depth;
        this.in_sx = opt.in_sx;
        this.in_sy = opt.in_sy;

        // optional
        this.sy = opt.sy != null ? opt.sy : this.sx;
        this.stride = opt.stride != null ? opt.stride : 2;
        this.pad = opt.pad != null ? opt.pad : 0; // amount of 0 padding to add around borders of input volume

        // computed
        this.out_depth = this.in_depth;
        this.out_sx = Math.floor((this.in_sx + this.pad * 2 - this.sx) / this.stride + 1);
        this.out_sy = Math.floor((this.in_sy + this.pad * 2 - this.sy) / this.stride + 1);
        this.layer_type = LayerType.POOL;

        // store switches for x,y coordinates for where the max comes from, for each output neuron
        this.switchx = Utils.zeros(this.out_sx * this.out_sy * this.out_depth);
        this.switchy = Utils.zeros(this.out_sx * this.out_sy * this.out_depth);
    }

    override public function forward(V:Vol, is_training:Bool):Vol {
        this.in_act = V;

        var A:Vol = new Vol(this.out_sx, this.out_sy, this.out_depth, [0]);

        var n:Int = 0; // a counter for switches

        for(d in 0 ... this.out_depth) {
            var x:Int = -this.pad;
            var y:Int = -this.pad;

            for (ax in 0 ... this.out_sx) {
                x += this.stride
                y = -this.pad;

                for(ay in 0 ... this.out_sy) {
                    y += this.stride;

                    // convolve centered at this particular location
                    var a:Int = -99999; // hopefully small enough ;\
                    var winx:Int = -1;
                    var winy:Int = -1;

                    for(fx in 0 ... this.sx) {
                        for(fy in 0 ... this.sy) {
                            var oy:Int = y + fy;
                            var ox:Int = x + fx;

                            if (oy >= 0 && oy < V.sy && ox >= 0 && ox < V.sx) {
                                var v:Float = V.get(ox, oy, d);

                                // perform max pooling and store pointers to where
                                // the max came from. This will speed up backprop
                                // and can help make nice visualizations in future

                                if (v > a) {
                                    a = v;
                                    winx = ox;
                                    winy = oy;
                                }
                            }
                        }
                    }

                    this.switchx[n] = winx;
                    this.switchy[n] = winy;
                    n++;

                    A.set(ax, ay, d, a);
                }
            }
        }

        this.out_act = A;
        return this.out_act;
    }

    override public function backward(y:Array<Float> = null):Null<Float> {
        // pooling layers have no parameters, so simply compute
        // gradient wrt data here

        var V:Vol = this.in_act;
        V.dw = Utils.zeros(V.w.length); // zero out gradient wrt data

        var A:Vol = this.out_act; // computed in forward pass

        var n:Int = 0;

        for (d in 0 ... this.out_depth) {
            var x:Int = -this.pad;
            var y:Int = -this.pad;

            for (ax in 0 ... this.out_sx) {
                x+=this.stride;
                y = -this.pad;

                for(ay in 0 ... this.out_sy) {
                    y += this.stride;

                    var chain_grad:Float = this.out_act.get_grad(ax, ay, d);

                    V.add_grad(this.switchx[n], this.switchy[n], d, chain_grad);
                    n++;
                }
            }
        }

        return null;
    }

    override public function toJSON():Dynamic {
        var json:Dynamic = {};
        json.sx = this.sx;
        json.sy = this.sy;
        json.stride = this.stride;
        json.in_depth = this.in_depth;
        json.out_depth = this.out_depth;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.layer_type = this.layer_type;
        json.pad = this.pad;
        return json;
    }

    override public function fromJSON(json:Dynamic):Void {
        this.out_depth = json.out_depth;
        this.out_sx = json.out_sx;
        this.out_sy = json.out_sy;
        this.layer_type = json.layer_type;
        this.sx = json.sx;
        this.sy = json.sy;
        this.stride = json.stride;
        this.in_depth = json.in_depth;
        this.pad = json.pad != null ? json.pad : 0; // backwards compatibility
        this.switchx = Utils.zeros(this.out_sx * this.out_sy * this.out_depth); // need to re-init these appropriately
        this.switchy = Utils.zeros(this.out_sx * this.out_sy * this.out_depth);
    }
}
