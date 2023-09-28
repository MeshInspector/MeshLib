window.onbeforeunload = (e) => {
    Module.ccall('emsForceSettingsSave', 'void', [], []);
    return "good";
};