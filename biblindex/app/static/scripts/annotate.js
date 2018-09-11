

function isOrContains(node, container) {
    while (node) {
        if (node === container) return true;
        node = node.parentNode;
    }
    return false;
}

function elementContainsSelection(el) {
  var sel = window.getSelection();
  if (sel.rangeCount > 0) {
    for (var i = 0; i < sel.rangeCount; ++i) {
      if (!isOrContains(sel.getRangeAt(i).commonAncestorContainer, el)) {
        return false;
      }
    }
    return true;
  }
  return false;
}

function getTextOfSelection(id) {
  if (elementContainsSelection(document.getElementById('bernard'))) {
    var endText = window.getSelection().getRangeAt(0).endContainer.wholeText;
    var nextText = $('#next').text();
    if (nextText.indexOf(endText) !== -1) {
      return window.getSelection().toString().replace(/(?:\r\n|\r|\n)+/g, ' ');
    }
  }
  return "";
}

function expandSelection(range) {
  if (range.collapsed) {
    return;
  }
  /** expand left */
  while (range.toString()[0].match(/\w/)) {
    range.setStart(range.startContainer, range.startOffset - 1);
  }
  range.setStart(range.startContainer, range.startOffset + 1);
  /** expand right */
  while (range.toString()[range.toString().length - 1].match(/\w/)) {
    range.setEnd(range.endContainer, range.endOffset + 1);
  }
  range.setEnd(range.endContainer, range.endOffset - 1);
}

function buildTextMeta(meta, text, id) {
  var p = text ? "<p class='text-center'><strong id='target'>" : "<p id='" + id + "'>";

  /** add text */
  for (var i=0; i<meta.length; i++) {
    p = p + meta[i]["word"] + " ";
  }

  /** close */
  if (text) {
    p = p + "</strong>";
  }
  p = p + "</p>";

  return p;
}

function renderItem(idx) {
  /** build bible */
  $('#bibleLink').empty();
  $('#bible').text(data[idx]["ref"]);
  $("#bibleLink").append("<a href='" + data[idx]["url"] + "' target='_blank'>" + data[idx]['url'] + "</a>");

  /** build bernard */
  var $bernard = $('#bernard');
  $bernard.empty();
  $bernard.append(buildTextMeta(data[idx]["textcontext"]["prev"], false, "prev"));
  $bernard.append(buildTextMeta(data[idx]["textdata"], true, "focus"));
  $bernard.append(buildTextMeta(data[idx]["textcontext"]["next"], false, "next"));  
}

$( document ).ready(function() {
  /** add event listiner to skip */
  $('#skip').on('click', function() {
    if (current === (data.length-1)) {
      $('#restartModal').modal();
    } else {
      current = current + 1;
    }
    renderItem(current);
  });

  /** add event listiner to save */
  $('#save').on('click', function() {

    /** save selection */
    var selection = getTextOfSelection('bernard');

    if (selection === "") {
      $('#selectionWarning').modal();
      return;
    } else {
      console.log("Saving: " + selection);
    }

    /** move to next */
    if (current === (data.length-1)) {
      $('#restartModal').modal();
    } else {
      current = current + 1;
    }
    renderItem(current);
  });

  /** add event listener to modal restart */
  $('#restart').on('click', function() {
    document.location.href='/annotate';
  });

  /** add selection logic */
  $('#bernard').on('mouseup', function() {
    expandSelection(window.getSelection().getRangeAt(0));
  });
  
  /** start */
  renderItem(current);  
});
