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