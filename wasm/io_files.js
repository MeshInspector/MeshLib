var open_files_dialog_popup = function (extensions) {
  var { overlay, popup } = createOverlayPopup('show_browse_dialog', "Select File", 400, 150);

  var file_selector_label = document.createElement('label');
  file_selector_label.setAttribute('for', 'FileSelectorTag');
  file_selector_label.setAttribute('style', 'position:absolute;top:50%;left:50%;transform:translate(-50%,50%);width: 120px;  height: 28px; border-radius: 4px;');
  file_selector_label.setAttribute('class', 'button');

  var file_selector_label_text = document.createElement('div');
  file_selector_label_text.innerHTML = "Browse...";
  file_selector_label_text.setAttribute('class', 'unselectable');
  file_selector_label_text.setAttribute('style', 'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size: 14px;  font-weight: 600;  font-stretch: normal;  font-style: normal;  line-height: normal;  letter-spacing: normal;  text-align: center;  color: #fff;');

  var file_selector = document.createElement('input');
  file_selector.setAttribute('type', 'file');
  file_selector.setAttribute('id', 'FileSelectorTag');
  file_selector.setAttribute('multiple', null);
  file_selector.setAttribute('onchange', 'open_files(event)');
  file_selector.setAttribute('accept', extensions);
  file_selector.setAttribute('style', 'display: none;');
  file_selector.setAttribute('align', 'center');

  file_selector_label.appendChild(file_selector_label_text);
  file_selector_label.appendChild(file_selector);

  popup.appendChild(file_selector_label);
  overlay.appendChild(popup);

  removeKeyboardEvents();
  document.body.appendChild(overlay);
}

var download_file_dialog_popup = function (defaultName, extensions) {

  var { overlay, popup } = createOverlayPopup('show_download_dialog', "Save File", 440, 232);

  var name_label = document.createElement('label');
  name_label.setAttribute('style', 'width: 144px;height: 20px;position: absolute;top: 86px;left: 62px;margin-left: 0px;font-size: 14px;color: #fff;');
  name_label.innerHTML = 'Name';
  name_label.setAttribute('class', 'unselectable');

  var name_selector = document.createElement('input');
  name_selector.setAttribute('type', 'text');
  name_selector.setAttribute('id', 'download_name');
  name_selector.setAttribute('style', 'position: absolute;top: 81px;left: 50%;transform: translate(-50%, 0px);background-color: #000;border-radius: 4px;width: 198px;height: 26px;border: solid 1px #5f6369;color: #fff;padding: 0px 0px;');

  name_selector.value = defaultName;

  var ext_label = document.createElement('label');
  ext_label.setAttribute('style', 'width: 59px;height: 20px;font-size: 14px;position: absolute;color: #fff;top: 131px;left: 38px;');
  ext_label.innerHTML = 'Extension';
  ext_label.setAttribute('class', 'unselectable');

  var list_item = document.createElement('select');
  list_item.setAttribute('id', 'download_ext');
  list_item.setAttribute('style', 'position: absolute;top: 125px;left: 50%;transform: translate(-50%, 0px);background-color: #000;border-radius: 4px;width: 200px;height: 28px;border: solid 1px #5f6369;color: #fff;padding: 0px 0px;');
  var splitExt = extensions.split(', ');
  for (var i = 0; i < splitExt.length; i++) {
    var option_el = document.createElement('option');
    option_el.setAttribute('value', splitExt[i]);
    option_el.innerHTML = splitExt[i];
    option_el.setAttribute('class', 'unselectable');
    list_item.appendChild(option_el);
  }

  var btn_save = document.createElement('input');
  btn_save.setAttribute('type', 'button');
  btn_save.setAttribute('value', 'Save');
  btn_save.setAttribute('style', 'position: absolute;width: 100px;height: 28px;top: 194px;left: 50%;transform: translate(-50%, -50%);border-radius: 4px;color: #fff;font-size: 14px;font-weight: 600;border: none;');
  btn_save.setAttribute('class', 'button');
  btn_save.setAttribute('onclick', 'Module.ccall(\'save_file\', \'number\', [\'string\'], [document.getElementById(\'download_name\').value + document.getElementById(\'download_ext\').value]),addKeyboardEvents(),document.getElementById(\'show_download_dialog\').remove()');

  popup.appendChild(name_label);
  popup.appendChild(name_selector);
  popup.appendChild(ext_label);
  popup.appendChild(list_item);
  popup.appendChild(btn_save);

  removeKeyboardEvents();
  document.body.appendChild(overlay);
}

var download_scene_dialog_popup = function () {
  var { overlay, popup } = createOverlayPopup('show_download_dialog', "Save Scene", 440, 190);

  var name_label = document.createElement('label');
  name_label.setAttribute('style', 'width: 144px;height: 20px;position: absolute;top: 94px;left: 62px;margin-left: 0px;font-size: 14px;color: #fff;');
  name_label.innerHTML = 'Name';
  name_label.setAttribute('class', 'unselectable');

  var name_selector = document.createElement('input');
  name_selector.setAttribute('type', 'text');
  name_selector.setAttribute('id', 'download_name');
  name_selector.setAttribute('style', 'position: absolute;top: 50%;transform: translate(20px, -50%);background-color: #000;border-radius: 4px;width: 200px;height: 28px;border: solid 1px #5f6369;color: #fff;');

  name_label.appendChild(name_selector);

  var btn_save = document.createElement('input');
  btn_save.setAttribute('type', 'button');
  btn_save.setAttribute('value', 'Save');
  btn_save.setAttribute('style', 'position: absolute;width: 100px;height: 28px;top: 156px;left: 50%;transform: translate(-50%, -50%);border-radius: 4px;color: #fff;font-size: 14px;font-weight: 600;border: none;');
  btn_save.setAttribute('onclick', 'Module.ccall(\'save_scene\', \'number\', [\'string\'], [document.getElementById(\'download_name\').value + \'.mru\']),addKeyboardEvents(),document.getElementById(\'show_download_dialog\').remove()');
  btn_save.setAttribute('class', 'button');

  popup.appendChild(name_label);
  popup.appendChild(btn_save);

  removeKeyboardEvents();
  document.body.appendChild(overlay);
}

var open_files = function (e) {
  if (!GLFW.active) {
    addKeyboardEvents();
    document.getElementById('show_browse_dialog').remove();
    return;
  }
  if (!e.target || !e.target.files || e.target.files.length == 0) {
    addKeyboardEvents();
    document.getElementById('show_browse_dialog').remove();
    return;
  }
  e.preventDefault();
  var filenames = _malloc(new Array(e.target.files.length * 4));
  var filenamesArray = [];
  var count = e.target.files.length;
  var written = 0;
  var drop_dir = ".use_open_files";
  FS.createPath("/", drop_dir);
  function save(file) {
    var path = "/" + drop_dir + "/" + file.name.replace(/\//g, "_");
    var reader = new FileReader();
    reader.onloadend = (e => {
      if (reader.readyState != 2) {
        ++written;
        out("failed to read opened file: " + file.name + ": " + reader.error);
        return;
      }
      var data = e.target.result;
      FS.writeFile(path, new Uint8Array(data));
      if (++written === count) {
        const res = Module.ccall('load_files', 'number', ['number', 'Int8Array'], [count, filenames]);
        for (var i = 0; i < filenamesArray.length; ++i) {
          _free(filenamesArray[i]);
        }
        _free(filenames);
      }
    });
    reader.readAsArrayBuffer(file);
    var filename = allocateUTF8(path);
    filenamesArray.push(filename);
    if (typeof GROWABLE_HEAP_I32 !== 'undefined')
      GROWABLE_HEAP_I32()[filenames + i * 4 >> 2] = filename;
    else
      HEAP32[filenames + i * 4 >> 2] = filename;
  }
  for (var i = 0; i < count; ++i) {
    save(e.target.files[i]);
  }
  addKeyboardEvents();
  document.getElementById('show_browse_dialog').remove();
  return false;
};

var save_file = function (filename) {
  var mime = "application/octet-stream";
  let content = FS.readFile(filename);

  var a = document.createElement('a');
  a.download = filename;
  a.href = URL.createObjectURL(new Blob([content], { type: mime }));
  a.style.display = 'none';

  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(a.href);
  }, 0);
};
