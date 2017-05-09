package convnethx;

import haxe.io.Float64Array;

class LayerMaxout extends Layer {

    public var group_size:Int;
    public var switches:Float64Array;

    public function new(opt:Opt) {
        super(opt);

        // required
        this.group_size = opt.group_size != null ? opt.group_size : 2;

        // computed
        this.out_sx = opt.in_sx;
        this.out_sy = opt.in_sy;
        this.out_depth = Math.floor(opt.in_depth / this.group_size);
        this.layer_type = LayerType.MAXOUT;

        this.switches = Utils.zeros(this.out_sx * this.out_sy * this.out_depth); // useful for backprop
    }

    override public function forward(V:Vol, is_training:Bool):Vol {
        this.in_act = V;

        var N:Int = this.out_depth;
        var V2:Vol = new Vol(this.out_sx, this.out_sy, this.out_depth, [0]);

        // optimization branch. If we're operating on 1D arrays we dont have
        // to worry about keeping track of x,y,d coordinates inside
        // input volumes. In convnets we do :(
        if(this.out_sx == 1 && this.out_sy == 1) {
            for(i in 0 ... N) {

                var ix:Int = i * this.group_size; // base index offset
                var a:Float = V.w[ix];
                var ai:Int = 0;

                for (j in 1 ... this.group_size) {
                    var a2:Float = V.w[ix + j];

                    if (a2 > a) {
                        a = a2;
                        ai = j;
                    }
                }

                V2.w[i] = a;
            this.switches[i] = ix + ai;
            }
        } else {
            var n:Int = 0; // counter for switches
            for(x in 0 ... V.sx) {
                for(y in 0 ... V.sy) {
                    for(i in 0 ... N) {
                        var ix:Int = i * this.group_size;
                        var a:Float = V.get(x, y, ix);
                        var ai:Int = 0;

                        for(j in 1 ... this.group_size) {
                            var a2:Float = V.get(x, y, ix + j);

                            if(a2 > a) {
                                a = a2;
                                ai = j;
                            }
                        }

                        V2.set(x, y, i, a);

                        this.switches[n] = ix + ai;
                        n++;
                    }
                }
            }
        }

        this.out_act = V2;

        return this.out_act;
    }

    override public function backward(y:Array<Float> = null):Float {
        var V:Vol = this.in_act; // we need to set dw of this
        var V2:Vol = this.out_act;
        var N:Int = this.out_depth;

        V.dw = Utils.zeros(V.w.length); // zero out gradient wrt data

        // pass the gradient through the appropriate switch
        if (this.out_sx == 1 && this.out_sy == 1) {
            for (i in 0 ... N) {
                var chain_grad:Float = V2.dw[i];
                V.dw[this.switches[i]] = chain_grad;
            }
        } else {
            // bleh okay, lets do this the hard way
            var n:Int = 0; // counter for switches
            for(x in 0 ... V2.sx) {
                for(y in 0 ... V2.sy) {
                    for(i in 0 ... N) {
                        var chain_grad:Float = V2.get_grad(x, y, i);
                        V.set_grad(x, y, this.switches[n], chain_grad);
                        n++;
                    }
                }
            }
        }
    }

    override public function getParamsAndGrads():Array<Dynamic> {

    }

    override public function toJSON():Dynamic {
        var json:Dynamic = {};
        json.out_depth = this.out_depth;
        json.out_sx = this.out_sx;
        json.out_sy = this.out_sy;
        json.layer_type = this.layer_type;
        json.group_size = this.group_size;
        return json;
    }

    override public function fromJSON(json:Dynamic):Void {
        this.out_depth = json.out_depth;
        this.out_sx = json.out_sx;
        this.out_sy = json.out_sy;
        this.layer_type = json.layer_type;
        this.group_size = json.group_size;
        this.switches = Utils.zeros(this.group_size);
    }
}
