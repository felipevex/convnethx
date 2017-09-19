package convnethx.trainer;

import convnethx.model.ParamsAndGradsValue;
import convnethx.type.LayerType;
import haxe.io.Float64Array;
class Trainer {

    public var net:Net;

    public var learning_rate:Float;
    public var l1_decay:Float;
    public var l2_decay:Float;
    public var batch_size:Int;
    public var momentum:Float;

    public var ro:Float;
    public var eps:Float;
    public var beta1:Float;
    public var beta2:Float;

    public var method:String;

    public var k:Int = 0; // iteration counter;
    public var gsum:Array<Float64Array> = []; // last iteration gradients (used for momentum calculations)
    public var xsum:Array<Float64Array> = []; // used in adam or adadelta

    public var regression:Bool;


    public function new(net:Net, options:TrainerOptions = null) {
        this.net = net;

        if (options == null) options = {};

        this.learning_rate = options.learning_rate != null ? options.learning_rate : 0.01;
        this.l1_decay = options.l1_decay != null ? options.l1_decay : 0.0;
        this.l2_decay = options.l2_decay != null ? options.l2_decay : 0.0;
        this.batch_size = options.batch_size != null ? options.batch_size : 1;
        this.method = options.method != null ? options.method : TrainerMethod.SDG; // sgd/adam/adagrad/adadelta/windowgrad/netsterov

        this.momentum = options.momentum != null ? options.momentum : 0.9;
        this.ro = options.ro != null ? options.ro : 0.95; // used in adadelta
        this.eps = options.eps != null ? options.eps : 1e-8; // used in adam or adadelta
        this.beta1 = options.beta1 != null ? options.beta1 : 0.9; // used in adam
        this.beta2 = options.beta2 != null ? options.beta2 : 0.999; // used in adam

        // check if regression is expected
        this.regression = this.net.layers[this.net.layers.length - 1].layer_type == LayerType.REGRESSION;
    }

    public function train(x:Vol, ?y:Int, ?volList:Array<Vol> = null) {

        var start:Float = Date.now().getTime();
        this.net.forward(x, true); // also set the flag that lets the net know we're just training
        var end:Float = Date.now().getTime();
        var fwd_time:Float = end - start;


        var start:Float = Date.now().getTime();
        var cost_loss:Null<Float> = this.net.backward(y);
        var l2_decay_loss:Float = 0.0;
        var l1_decay_loss:Float = 0.0;
        var end:Float = Date.now().getTime();
        var bwd_time:Float = end - start;

        if (this.regression && volList == null)
            throw "Warning: a regression net requires an array as training output vector.";


        this.k++;

        if (this.k % this.batch_size == 0) {
            var pglist:Array<ParamsAndGradsValue> = this.net.getParamsAndGrads();

            // initialize lists for accumulators. Will only be done once on first iteration
            if(this.gsum.length == 0 && (this.method != TrainerMethod.SDG || this.momentum > 0.0)) {
                // only vanilla sgd doesnt need either lists
                // momentum needs gsum
                // adagrad needs gsum
                // adam and adadelta needs gsum and xsum

                for (i in 0 ... pglist.length) {
                    this.gsum.push(Utils.zeros(pglist[i].params.length));

                    switch (this.method) {
                        case TrainerMethod.ADAM | TrainerMethod.ADADELTA : {
                            this.xsum.push(Utils.zeros(pglist[i].params.length));
                        }
                        case _ : {
                            this.xsum.push(new Float64Array(0)); // conserve memory
                        }
                    }
                }
            }

            // perform an update for all sets of weights
            for (i in 0 ... pglist.length) {
                var pg:ParamsAndGradsValue = pglist[i]; // param, gradient, other options in future (custom learning rate etc)

                var p:Float64Array = pg.params;
                var g:Float64Array = pg.grads;

                // learning rate for some parameters.
                var l2_decay_mul:Float = pg.l2_decay_mul != null ? pg.l2_decay_mul : 1.0;
                var l1_decay_mul:Float = pg.l1_decay_mul != null ? pg.l1_decay_mul : 1.0;
                var l2_decay:Float = this.l2_decay * l2_decay_mul;
                var l1_decay:Float = this.l1_decay * l1_decay_mul;


                var plen:Int = p.length;

                for (j in 0 ... plen) {

                    l2_decay_loss += l2_decay * p[j] * p[j] / 2; // accumulate weight decay loss
                    l1_decay_loss += l1_decay * Math.abs(p[j]);

                    var l1grad:Float = l1_decay * (p[j] > 0 ? 1 : -1);
                    var l2grad:Float = l2_decay * (p[j]);

                    var gij:Float = (l2grad + l1grad + g[j]) / this.batch_size; // raw batch gradient

                    var gsumi:Float64Array = this.gsum[i];
                    var xsumi:Float64Array = this.xsum[i];

                    switch (this.method) {
                        case TrainerMethod.ADAM : {
                            // adam update
                            gsumi[j] = gsumi[j] * this.beta1 + (1 - this.beta1) * gij; // update biased first moment estimate
                            xsumi[j] = xsumi[j] * this.beta2 + (1 - this.beta2) * gij * gij; // update biased second moment estimate

                            var biasCorr1:Float = gsumi[j] * (1 - Math.pow(this.beta1, this.k)); // correct bias first moment estimate
                            var biasCorr2:Float = xsumi[j] * (1 - Math.pow(this.beta2, this.k)); // correct bias second moment estimate
                            var dx:Float = - this.learning_rate * biasCorr1 / (Math.sqrt(biasCorr2) + this.eps);

                            p[j] += dx;
                        }

                        case TrainerMethod.ADAGRAD : {
                            // adagrad update
                            gsumi[j] = gsumi[j] + gij * gij;
                            var dx:Float = - this.learning_rate / Math.sqrt(gsumi[j] + this.eps) * gij;
                            p[j] += dx;
                        }

                        case TrainerMethod.WINDOWGRAD : {
                            // this is adagrad but with a moving window weighted average
                            // so the gradient is not accumulated over the entire history of the run.
                            // it's also referred to as Idea #1 in Zeiler paper on Adadelta. Seems reasonable to me!
                            gsumi[j] = this.ro * gsumi[j] + (1-this.ro) * gij * gij;
                            var dx:Float = - this.learning_rate / Math.sqrt(gsumi[j] + this.eps) * gij; // eps added for better conditioning
                            p[j] += dx;
                        }

                        case TrainerMethod.ADADELTA : {
                            gsumi[j] = this.ro * gsumi[j] + (1 - this.ro) * gij * gij;
                            var dx:Float = - Math.sqrt((xsumi[j] + this.eps) / (gsumi[j] + this.eps)) * gij;
                            xsumi[j] = this.ro * xsumi[j] + (1 - this.ro) * dx * dx; // yes, xsum lags behind gsum by 1.
                            p[j] += dx;
                        }

                        case TrainerMethod.NETSTEROV : {
                            var dx:Float = gsumi[j];
                            gsumi[j] = gsumi[j] * this.momentum + this.learning_rate * gij;
                            dx = this.momentum * dx - (1.0 + this.momentum) * gsumi[j];
                            p[j] += dx;
                        }

                        case _: {
                            // assume SGD
                            if (this.momentum > 0.0) {
                                // momentum update
                                var dx:Float = this.momentum * gsumi[j] - this.learning_rate * gij; // step
                                gsumi[j] = dx; // back this up for next iteration of momentum
                                p[j] += dx; // apply corrected gradient
                            } else {
                                // vanilla sgd
                                p[j] +=  - this.learning_rate * gij;
                            }
                        }
                    }

                    g[j] = 0.0; // zero out gradient so that we can begin accumulating anew
                }
            }
        }

        // appending softmax_loss for backwards compatibility, but from now on we will always use cost_loss
        // in future, TODO: have to completely redo the way loss is done around the network as currently
        // loss is a bit of a hack. Ideally, user should specify arbitrary number of loss functions on any layer
        // and it should all be computed correctly and automatically.

        return {
            fwd_time: fwd_time,
            bwd_time: bwd_time,
            l2_decay_loss: l2_decay_loss,
            l1_decay_loss: l1_decay_loss,
            cost_loss: cost_loss,
            softmax_loss: cost_loss,
            loss: cost_loss + l1_decay_loss + l2_decay_loss
        };
    }
}
