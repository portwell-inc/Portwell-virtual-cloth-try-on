var countDown = {
    dfCountTime:10,//預設倒數秒數
    blinkingTime:5,//預設幾秒後開始閃爍
    countdownId:null,//setInterval ID
    countTime:-1,//計算用
    /*
     *
     * @param countId 顯示倒數數字的Element id
     * @param endFun 計時結束要做的function
     * @param dfCountTime 倒數秒數
     * @param blinkingTime 幾秒後開始閃爍
     */
    init:function(countId, endFun, dfCountTime, blinkingTime) {
 
        if (dfCountTime != undefined)
            this.dfCountTime = dfCountTime;
 
        if (blinkingTime != undefined)
            this.blinkingTime = blinkingTime;
 
        this.endFun = endFun;
        this.countId = countId;
        this.countTarget = document.getElementById(this.countId);
        this.countTarget.innerHTML = "";
    },
    /**
     * 到數計時
     */
    _countdown:function() {
 
        if (this.countTime == -1) {
            this.countTime = this.dfCountTime;
        }
 
        //小於10秒時會開始閃爍
        if (this.countTime <= this.blinkingTime) {
            this.countTarget.innerHTML = "" + (this.countTime--) + "";
            setTimeout(function() {
                countDown.countTarget.innerHTML = "";
            }, 500);
        } else {
            this.countTarget.innerHTML = "" + (this.countTime--) + "";
        }
 
        //時間到了
        if (this.countTime < 0) {
            this.endFun();//
            this.stop();
        }
 
    },
    /**
     * 停止到數
     */
    stop:function() {
        if (this.countdownId != null && this.countdownId != undefined) {
            clearInterval(this.countdownId);
        }
        this.countTime = -1;
        if (this.countTarget != undefined) {
            this.countTarget.innerHTML = "";
        }
    },
    /**
     * 開始
     */
    start:function() {
        //倒數計時
        this.countdownId = setInterval(function() {
            countDown._countdown();
        }, 1000);
    }
}