var append_disclamer = function () {
  if (window.location.href.indexOf("RMISingle") < 0)
    return;

  var popup = createPopup('single_thread_disclamer_dialog', '', 600, 60);
  popup.setAttribute('id', 'single_thread_disclamer_dialog');

  var logo = document.createElement('img');
  logo.setAttribute('src', 'Fx-Browser-icon-fullColor.svg');
  logo.setAttribute('style', 'position: absolute;top: 18px;left: 85px;width:24px;height:24px');

  var disclamer = document.createElement('p');
  disclamer.setAttribute('style', 'color: #fff;position: absolute;top: 8px;left: 131px;font-size: 14px;font-weight: 600;');
  disclamer.innerHTML = 'For better experience on MacOS use Firefox browser';

  popup.appendChild(logo);
  popup.appendChild(disclamer);
  document.body.appendChild(popup);
}

append_disclamer();