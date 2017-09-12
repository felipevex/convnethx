package convnethx;

import convnethx.model.DefVolume;
import convnethx.Utils;
import haxe.io.Float64Array;

class Vol {

    /**
    * Vol is the basic building block of all data in a net.
    * it is essentially just a 3D volume of numbers, with a
    * width (sx), height (sy), and depth (depth).
    * it is used to hold data for all filters, all volumes,
    * all weights, and also stores all gradients w.r.t.
    * the data. constantValue is optionally a value to initialize the volume
    * with. If constantValue is missing, fills the Vol with random numbers.
    **/

    public var sx:Int;
    public var sy:Int;
    public var depth:Int;

    public var w:Float64Array;
    public var dw:Float64Array;


    public function new(?volumeValues:Array<Float>, ?sx:Int, ?sy:Int, ?depth:Int, ?constantValue:Float) {
        if (volumeValues != null && volumeValues.length > 0) {

            this.sx = 1;
            this.sy = 1;
            this.depth = volumeValues.length;

            this.w = Utils.zeros(this.depth);
            this.dw = Utils.zeros(this.depth);

            for (i in 0 ... volumeValues.length) this.w.set(i, volumeValues[i]);

        } else {

            if (sx == null || sy == null) throw "You must input sx or sy value";

            this.sx = sx;
            this.sy = sy;
            this.depth = depth;

            var n:Int = this.sx * this.sy * this.depth;

            this.w = Utils.zeros(n);
            this.dw = Utils.zeros(n);

            if (constantValue == null) {
                // weight normalization is done to equalize the output
                // variance of every neuron, otherwise neurons with a lot
                // of incoming connections have outputs of larger variance

                var scale:Float = Math.sqrt(1.0 / (sx * sy * depth));
                for (i in 0 ... n) this.w.set(i, Utils.randn(0, scale));

            } else {
                for (i in 0 ... this.w.length) this.w.set(i, constantValue);

            }
        }
    }

    inline private function getIndex(x:Int, y:Int, depth:Int):Int return ((this.sx * y) + x) * this.depth + depth;

    public function get(x:Int, y:Int, depth:Int):Float return this.w[this.getIndex(x, y, depth)];
    public function set(x:Int, y:Int, depth:Int, value:Float):Void this.w[this.getIndex(x, y, depth)] = value;
    public function add(x:Int, y:Int, depth:Int, value:Float):Void this.w[this.getIndex(x, y, depth)] += value;

    public function get_grad(x:Int, y:Int, depth:Int):Float return this.dw[this.getIndex(x, y, depth)];
    public function set_grad(x:Int, y:Int, depth:Int, value:Float):Void this.dw[this.getIndex(x, y, depth)] = value;
    public function add_grad(x:Int, y:Int, depth:Int, value:Float):Void this.dw[this.getIndex(x, y, depth)] += value;

    public function cloneAndZero():Vol return new Vol(this.sx, this.sy, this.depth, 0);

    public function clone():Vol {
        var clone:Vol = new Vol(this.sx, this.sy, this.depth, 0);
        for (i in 0 ... this.w.length) clone.w.set(i, this.w.get(i));
        return clone;
    }

    public function setConst(constValue:Float):Void for (i in 0 ... this.w.length) this.w[i] = constValue;
    public function addFrom(vol:Vol):Void for (i in 0 ... this.w.length) this.w[i] += vol.w[i];
    public function addFromScaled(vol:Vol, scale:Float):Void for (i in 0 ... this.w.length) this.w[i] += (scale * vol.w[i]);

    // todo: we may want to only save d most significant digits to save space
    public function toJSON():DefVolume {
        var json:DefVolume = {
            sx : this.sx,
            sy : this.sy,
            depth : this.depth,
            w : Utils.convertToFloatArray(this.w)
        };

        return json;
        // we wont back up gradients to save space
    }

    public function fromJSON(json:DefVolume):Void {
        this.sx = json.sx;
        this.sy = json.sy;
        this.depth = json.depth;
        this.w = Utils.convertToFloat64Array(json.w);
        this.dw = Utils.zeros(this.w.length);
    }
}
