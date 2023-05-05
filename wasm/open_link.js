var open_link = function (link) {
    var { overlay, popup } = createOverlayPopup('open_link_dialog', '', 200, 100);

    var linkItem = document.createElement('a');
    linkItem.setAttribute('href', link);
    linkItem.innerHTML = 'Follow link';
    linkItem.setAttribute('target', '_blank');
    linkItem.setAttribute('onclick', 'document.getElementById(\'open_link_dialog\').remove(),addKeyboardEvents()');
    linkItem.setAttribute('style', 
    'position: absolute;\
    width: 100px;height: 28px;\
    top: 50%;left: 50%;\
    transform: translate(-50%, -50%);\
    border-radius: 4px;color: #fff;\
    font-size: 14px;\
    font-weight: 600;\
    border: none;\
    display: flex;\
    align-items: center;\
    justify-content: center;');
    linkItem.setAttribute('class', 'button');

    popup.appendChild(linkItem);
    overlay.appendChild(popup);

    removeKeyboardEvents();
    document.body.appendChild(overlay);
}