package convnethx;

import haxe.io.Float64Array;

class Utils {

    private static var return_v:Bool = false;
    private static var v_val:Float = 0;

    private static function gaussRandom():Float {

        if (return_v) {
            return_v = false;
            return v_val;
        }

        var u:Float = 2 * Math.random() - 1;
        var v:Float = 2 * Math.random() - 1;
        var r:Float = u * u + v * v;

        if(r == 0 || r > 1) return gaussRandom();

        var c:Float = Math.sqrt(-2 * Math.log(r) / r);

        v_val = v * c; // cache this
        return_v = true;

        return u * c;
    }


    public static function randf(a:Float, b:Float):Float {
        return Math.random() * (b - a) + a;
    }

    public static function randi(a:Int, b:Int):Int {
        return Math.floor(Math.random() * (b - a) + a);
    }

    public static function randn(mu:Float, std:Float):Float{
        return mu + gaussRandom() * std;
    }

    /**
    * Array utilities
    **/
    public static function zeros(n:Int):Float64Array {
        return new Float64Array(n);
    }

    public static function convertToFloat64Array(values:Array<Float>):Float64Array {
        var result:Float64Array = zeros(values.length);

        for (i in 0 ... values.length) {
            result[i] = values[i];
        }

        return result;
    }

    public static function convertToFloatArray(values:Float64Array):Array<Float> {
        var result:Array<Float> = [];

        for (i in 0 ... values.length) {
            result.push(values[i]);
        }

        return result;
    }

    /**
    * return max and min of a given non-empty array.
    **/
    public static function maxmin(w:Array<Float>):{maxi: Int, maxv: Float, mini: Int, minv: Float, dv: Float} {

        if(w.length == 0) {
            return {};
        }

        var maxv:Float = w[0];
        var minv:Float = w[0];

        var maxi:Int = 0;
        var mini:Int = 0;

        var n = w.length;

        for (i in 0 ... w.length) {
            if(w[i] > maxv) {
                maxv = w[i];
                maxi = i;
            }

            if(w[i] < minv) {
                minv = w[i];
                mini = i;
            }



        }

        return {
            maxi: maxi,
            maxv: maxv,
            mini: mini,
            minv: minv,
            dv:maxv-minv
        };
    }

    /**
    * create random permutation of numbers, in range [0...n-1]
    **/
    public static function randperm(n:Int):Array<Int> {
        var i:Int = n;
        var j:Int = 0;
        var temp:Int;

        var array:Array<Int> = [];

        for (q in 0 ... n) array.push(q);

        while (i >= 0) {
            j = Math.floor(Math.random() * (i + 1));

            temp = array[i];
            array[i] = array[j];
            array[j] = temp;

            i--;
        }

        return array;
    }

    /**
    * sample from list lst according to probabilities in list probs
    * the two lists are of same size, and probs adds up to 1
    **/
    public static function weightedSample(lst:Array<Float>, probs:Array<Float>) {
        var p:Float = randf(0, 1.0);

        var cumprob:Float = 0.0;

        for (k in 0 ... lst.length) {
            cumprob += probs[k];
            if (p < cumprob) return lst[k];
        }

        return 0;
    }

    public static function arrUnique(arr:Array<Dynamic>):Array<Dynamic> {
        var b:Array<Dynamic> = [];

        for (element in arr) {
            if (!arrContains(b, element)) b.push(element);
        }

        return b;
    }

    public static function arrContains(arr:Array<Dynamic>, elt:Dynamic):Bool {
        for (element in arr) {
            if (element == elt) return true;
        }

        return false;
    }

    /**
    * syntactic sugar function for getting default parameter values
    **/
    public static function getopt(opt:Opt, field_name:Array<String>, default_value:Dynamic):Dynamic {
        var result:Dynamic = default_value;

        for (field in field_name) {
            var tempResult:Dynamic = Reflect.field(opt, field);
            if (tempResult != null) result = tempResult;
        }

        return result;
    }

    public static function assert(condition:Bool, message:String):Void {
        if (!condition) {
            message = message || "Assertion failed";

            throw message;
        }
    }
}