package convnethx;

import convnethx.Utils;
import haxe.io.Float64Array;

class Vol {

    /**
    * Vol is the basic building block of all data in a net.
    * it is essentially just a 3D volume of numbers, with a
    * width (sx), height (sy), and depth (depth).
    * it is used to hold data for all filters, all volumes,
    * all weights, and also stores all gradients w.r.t.
    * the data. c is optionally a value to initialize the volume
    * with. If c is missing, fills the Vol with random numbers.
    **/

    public var sx:Int;
    public var sy:Int;
    public var depth:Int;

    public var w:Float64Array;
    public var dw:Float64Array;


    public function new(sx:Int, sy:Int, depth:Int, c:Array<Float> = null) {

        this.sx = sx;
        this.sy = sy;
        this.depth = depth;

        var n:Int = this.sx * this.sy * this.depth;

        this.w = Utils.zeros(n);
        this.dw = Utils.zeros(n);

        if (c == null || c.length == 0) {
            // weight normalization is done to equalize the output
            // variance of every neuron, otherwise neurons with a lot
            // of incoming connections have outputs of larger variance

            var scale:Float = Math.sqrt(1 / (sx * sy * depth));

            for (i in 0 ... n) this.w[i] = Utils.randn(0, scale);

        } else {
            if (c.length == n)
                for (i in 0 ... n) this.w[i] = c[i];
            else
                for (i in 0 ... n) this.w[i] = c[0];
        }
    }

    public static function generate1DVol(values:Array<Float>):Vol {
        return new Vol(1, 1, values.length, values);
    }

    inline private function getIndex(x:Int, y:Int, d:Int):Int {
        return ((this.sx * y) + x) * this.depth + d;
    }

    public function get(x:Int, y:Int, d:Int):Float {
        var index:Int = this.getIndex(x, y, d);
        return this.w[index];
    }

    public function set(x:Int, y:Int, d:Int, value:Float):Float {
        var index:Int = this.getIndex(x, y, d);
        this.w[index] = value;
    }

    public function add(x:Int, y:Int, d:Int, value:Float):Float {
        var index:Int = this.getIndex(x, y, d);
        this.w[index] += value;
    }

    public function get_grad(x:Int, y:Int, d:Int):Float {
        var index:Int = this.getIndex(x, y, d);
        return this.dw[index];
    }

    public function set_grad(x:Int, y:Int, d:Int, value:Float):Float {
        var index:Int = this.getIndex(x, y, d);
        this.w[index] = value;
    }

    public function add_grad(x:Int, y:Int, d:Int, value:Float):Float {
        var index:Int = this.getIndex(x, y, d);
        this.w[index] += value;
    }

    public function cloneAndZero():Vol {
        return new Vol(this.sx, this.sy, this.depth, [0]);
    }

    public function clone():Vol {
        var values:Array<Float> = [];

        for (i in 0 ... this.w.length) {
            values.push(this.w.get(i));
        }

        return new Vol(this.sx, this.sy, this.depth, values);
    }

    public function addFrom(vol:Vol):Void {
        for (i in 0 ... this.w.length) {
            this.w[i] += vol.w[i];
        }
    }

    public function addFromScaled(vol:Vol, a:Float):Void {
        for (i in 0 ... this.w.length) {
            this.w[i] += a * vol.w[i];
        }
    }

    public function setConst(a:Float) {
        for (i in 0 ... this.w.length) {
            this.w[i] = a;
        }
    }

    public function toJSON():Dynamic {
        // todo: we may want to only save d most significant digits to save space

        var json:Dynamic = {}
        json.sx = this.sx;
        json.sy = this.sy;
        json.depth = this.depth;
        json.w = Utils.convertToFloatArray(this.w);

        return json;
        // we wont back up gradients to save space
    }

    public function fromJSON(json:Dynamic):Void {
        this.sx = json.sx;
        this.sy = json.sy;
        this.depth = json.depth;
        this.w = Utils.convertToFloat64Array(json.w);
        this.dw = Utils.zeros(this.w.length);
    }
}
