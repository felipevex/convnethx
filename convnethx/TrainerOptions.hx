package convnethx;

typedef TrainerOptions = {
    @:optional var learning_rate:Float;
    @:optional var l1_decay:Float;
    @:optional var l2_decay:Float;
    @:optional var batch_size:Int;
    @:optional var method:String;
    @:optional var momentum:Float;
    @:optional var ro:Float;
    @:optional var eps:Float;
    @:optional var beta1:Float;
    @:optional var beta2:Float;
}
