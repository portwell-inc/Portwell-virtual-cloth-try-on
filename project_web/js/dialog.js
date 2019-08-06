(function($) {
    var h = [];
    h.push("<div class='cnfrm-block'>");
    h.push("<div class='cnfrm-msg'></div>");
    h.push("<input type='button' class='cnfrm-yes' />");
    h.push("<input type='button' class='cnfrm-no' />");
    h.push("</div>");
    var html = h.join("");

    $.kendoConfirm = function(title, msg, yesText, noText) {
        var $div = $(html);
        $div.find(".cnfrm-msg").text(msg);
        $div.find(".cnfrm-yes").val(yesText || "Yes");
        $div.find(".cnfrm-no").val(noText || "No");
        var win = $div.kendoWindow({
            title: title || "Confirmation",
            resizable: false,
            modal: true,
            deactivate: function() {
                this.destroy(); //remove itself after close
            }
        }).data("kendoWindow");
        win.center().open();
        var dfd = $.Deferred();
        $div.find(":button").click(function() {
            win.close();
            if (this.className == "cnfrm-yes")
                dfd.resolve();
            else
                dfd.reject();
        });
        return dfd.promise();
    };
})(jQuery);

$(function() {
    $("#btnTest").click(function() {
        var dfd =
            $.kendoConfirm(
                "Please confirm...",
                "Are you sure to delete it?",
                "Yeeees", "No no no");
        dfd.done(function() { //按下Yes時
                alert("You are sure");
            })
            .fail(function() { //按下No時
                alert("You are not sure");
            });
    });
});