var save_config = function (config_str) {
    localStorage.setItem('config', config_str);
}

var load_config = function () {
    localStorage.getItem('config');
}