var resizeCallBack = function () {
  if (!hasMouse() && window.devicePixelRatio > 2)
    window.devicePixelRatio = 2;
  document.getElementById("canvas").setAttribute('style', 'width: 100%;height: 100%;');
  var rect = window.document.body.getBoundingClientRect();
  Module.ccall('resizeEmsCanvas', 'number', ['number', 'number'], [window.innerWidth, window.innerHeight]);
  document.getElementById("canvas").setAttribute('style', 'width: 100%;height: 100%;');
  // calc twice (for instant update)
  var rect = window.document.body.getBoundingClientRect();
  Module.ccall('resizeEmsCanvas', 'number', ['number', 'number'], [rect.width, rect.height]);
  document.getElementById("canvas").setAttribute('style', 'width: 100%;height: 100%;');
}

var registerResize = function () {
  resizeCallBack();
  window.addEventListener('resize', resizeCallBack);
  window.addEventListener('orientationchange', resizeCallBack);
}
