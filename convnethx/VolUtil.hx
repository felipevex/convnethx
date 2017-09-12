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

    #if js

    /**
    * img is a DOM element that contains a loaded image
    * returns a Vol of size (W, H, 4). 4 is for RGBA
    **/
    public static function img_to_vol(img:js.html.Image, convert_grayscale:Bool = false):Vol {

        var canvas:js.html.CanvasElement = js.Browser.document.createCanvasElement();
        canvas.width = img.width;
        canvas.height = img.height;

        var ctx:Dynamic = canvas.getContext("2d");

        // due to a Firefox bug
        try {
            ctx.drawImage(img, 0, 0);
        } catch (e) {
            if (e.name == "NS_ERROR_NOT_AVAILABLE") {

                // sometimes happens, lets just abort
                return null;

            } else {
                throw e;
            }
        }

        var img_data:Dynamic = null;

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
        var p = img_data.data;
        var W:Int = img.width;
        var H:Int = img.height;

        var pv:Array<Float> = []

        for(i in 0 ... p.length) {
            pv.push(p[i] / 255 - 0.5); // normalize image pixels to [-0.5, 0.5]
        }

        var x:Vol = new Vol(W, H, 4, 0); //input volume (image)
        x.w = pv;

        if (convert_grayscale) {

            // flatten into depth=1 array
            var x1:Vol = new Vol(W, H, 1, 0.0);

               for(i in 0 ... W) {
                   for(j in 0 ... H) {
                       x1.set(i, j, 0, x.get(i, j, 0));
                   }
               }

            x = x1;
        }

        return x;
    }
    #end

}
