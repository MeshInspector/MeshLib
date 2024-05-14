var is_ios = function () {
    // ios
    if ([
        'iPad Simulator',
        'iPhone Simulator',
        'iPod Simulator',
        'iPad',
        'iPhone',
        'iPod'
    ].includes(navigator.platform)
        // iPad on iOS 13 detection
        || (navigator.userAgent.includes("Mac") && "ontouchend" in document))
        return true;
    return false;
}

var is_mac = function () {
    if (is_ios())
        return false;
    return navigator.platform.toUpperCase().indexOf('MAC') >= 0;
}

var is_safari = function(){
    return /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
}

var is_android = function () {
    return /Android/i.test(navigator.userAgent);
}

var is_mobile = function () {
    return is_android() || is_ios();
}
