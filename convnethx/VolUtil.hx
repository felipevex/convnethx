package convnethx;

/**
* Volume utilities
* intended for use with data augmentation
* crop is the size of output
* dx,dy are offset wrt incoming volume, of the shift
* fliplr is boolean on whether we also want to flip left<->right
**/

class VolUtil {

    public function augment(V:Vol, crop:Int, dx:Null<Float> = null, dy:Null<Float> = null, fliplr:Bool = false):Vol {
        // note assumes square outputs of size crop x crop

        if (dx == null) dx = Utils.randi(0, V.sx - crop);
        if (dy == null) dy = Utils.randi(0, V.sy - crop);

        // randomly sample a crop in the input volume
        var W:Vol;

        if (crop != V.sx || dx != 0 || dy != 0) {
            W = new Vol(crop, crop, V.depth, 0.0);

            for (x in 0 ... crop) {
                for (y in 0 ... crop) {

                    if( x + dx < 0 || x + dx >= V.sx || y + dy < 0 || y + dy >= V.sy) continue; // oob

                    for(d in 0 ... V.depth) {
                        W.set(x, y, d, V.get(x + dx, y + dy, d)); // copy data over
                    }
                }
            }
        } else {
            W = V;
        }

        if (fliplr) {
            // flip volume horziontally

            var W2:Vol = W.cloneAndZero();

            for (x in 0 ... W.sx) {
                for (y in 0 ... W.sy) {
                    for (d in 0 ... W.depth) {
                        W2.set(x, y, d, W.get(W.sx - x - 1, y, d)); // copy data over
                    }
                }
            }

            W = W2; //swap
        }

        return W;
    }

    public static function imageToVol(width:Int, height:Int, RGBA:Array<Int>, ?convertToGrayscale:Bool = false):Vol {
        if (width == 0 || height == 0 || width * height != Math.floor(RGBA.length/4)) throw "Wrong image size";

        if (convertToGrayscale) {
            var grayFloat:Array<Float> = [];

            for (i in 0 ... (width * height)) {
                // 0.21 R + 0.72 G + 0.07 B.
                var r:Float = RGBA[i + 0] * 0.21;
                var g:Float = RGBA[i + 1] * 0.72;
                var b:Float = RGBA[i + 2] * 0.07;
                var a:Float = RGBA[i + 3] / 255;

                var g:Float = Math.floor((r + g + b) * a);

                grayFloat.push(g / 255 - 0.5);
            }

            var vol:Vol = new Vol(width, height, 1, 0);
            vol.w = Utils.convertToFloat64Array(grayFloat);

            return vol;

        } else {
            // color volume
            var colorFloat:Array<Float> = [for(value in RGBA) {value / 255 - 0.5}]; // normalize image pixels to [-0.5, 0.5]

            var vol:Vol = new Vol(width, height, 4, 0);
            vol.w = Utils.convertToFloat64Array(colorFloat);

            return vol;
        }

    }


    #if js

    /**
    * img is a DOM element that contains a loaded image
    * returns a Vol of size (W, H, 4). 4 is for RGBA
    **/
    public static function img_to_vol(img:js.html.Image, convertToGrayscale:Bool = false):Vol {
        var canvas:js.html.CanvasElement = js.Browser.document.createCanvasElement();
        canvas.width = img.width;
        canvas.height = img.height;

        var ctx:js.html.CanvasRenderingContext2D = canvas.getContext2d();

        // due to a Firefox bug
        try {
            ctx.drawImage(img, 0, 0);
        } catch (e:Dynamic) {
            if (e.name == "NS_ERROR_NOT_AVAILABLE") {

                // sometimes happens, lets just abort
                return null;

            } else {
                throw e;
            }
        }

        var img_data:js.html.ImageData = null;

        try {
            img_data = ctx.getImageData(0, 0, canvas.width, canvas.height);
        } catch (e) {
            if(e.name == 'IndexSizeError') {
                return null; // not sure what causes this sometimes but okay abort
            } else {
                throw e;
            }
        }

        // prepare the input: get pixels and normalize them
        var data:Array<Int> = [for (value in img_data.data) {value}];

        return imageToVol(img.width, img.height, data, convertToGrayscale);
    }
    #end

}
