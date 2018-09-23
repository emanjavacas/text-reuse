

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

function getSelectionData(id) {
  if (elementContainsSelection(document.getElementById('bernard'))) {
    var sel = window.getSelection();
    var output = {
      "text": sel.toString().replace(/(?:\r\n|\r|\n)+/g, ' '),
      "span": getSelectionSpan(sel)
    };
    var hasTarget, hasSomethingElse;
    var selectedNodes = sel.getRangeAt(0).cloneContents().childNodes;
    for (var i=0; i<selectedNodes.length; i++) {
      if (selectedNodes[i].id === "target") { hasTarget = true; }
      else if (selectedNodes[i].id === "prev" || selectedNodes[i].id === "next") {
	hasSomethingElse = true;
      }
    }
    if (hasTarget && hasSomethingElse) { return output; }
  }
  return {};
};

function getSelectionSpan(selection) {
  var childNodes = selection.getRangeAt(0).cloneContents().childNodes;
  if (childNodes.length < 2 || childNodes.length > 3) {
    console.log("I got too few or too many nodes, ", childNodes.length);
    return {};
  }
  var prev, next = 0;
  var j = 0;
  for (var i=0; i<childNodes.length; i++) {
    if (childNodes[i].id === "prev") {
      if (childNodes[i].length === 0) {
	console.log("I got too few child nodes for 'prev'");
	return {};
      }
      prev = childNodes[i].childNodes[j];
      while (!prev.hasAttribute("data-item")) {
	j++;
	prev = childNodes[i].childNodes[j];
      }
      j = 0;
    } else if (childNodes[i].id === "next") {
      if (childNodes[i].length === 0) {
	console.log("I got too few child nodes for 'next'");
	return {};
      }
      next = childNodes[i].childNodes[childNodes[i].childNodes.length-1-j];
      while(!next.hasAttribute("data-item")) {
	j++;
	console.log("up", next);
	next = childNodes[i].childNodes[childNodes[i].childNodes.length-1-j];
      }
    }
  }
  /** normalize and fix weird stuff with empty nodes */
  if (prev) {
    prevNumber = Number(prev.getAttribute("data-item"));
    if (prev.textContent === "") {
      prevNumber = Math.max(prevNumber - 1, 0);
    }
    prev = prevNumber;
  }
  if (next) {
    nextNumber = Number(next.getAttribute("data-item"));
    if (next.textContent === "") {
      nextNumber = Math.max(nextNumber - 1, 0);
    }
    next = nextNumber;
  }
  return {"prev": prev, "next": next};
};

function expandSelection(range) {
  if (range.collapsed) { return; }

  /** expand left */
  while (range.toString()[0].match(/\w/) && (range.startOffset > 0)) {
    range.setStart(range.startContainer, range.startOffset - 1);
  }

  /** expand right */
  while (range.toString()[range.toString().length - 1].match(/\w/)
	 && range.endOffset < range.endContainer.length) {
    range.setEnd(range.endContainer, range.endOffset + 1);
  }
}

function buildTextMeta(meta, text, id) {
  var p = text ?
      "<span class='text-center' id='" + id + "'><strong>" :
      "<span id='" + id + "'>";

  /** add text */
  for (var i=0; i<meta.length; i++) {
    var childId;
    switch (id) {
      case "prev": childId = meta.length - i; break;
      case "target": childId = 0; break;
      case "next": childId = i + 1; break;
    }
    p = p + "<span data-item='" + childId + "'>"
      + meta[i]["word"] + "</span>" + "<span> </span>";
  }

  /** close */
  if (text) {
    p = p + "</strong>";
  }
  p = p + "</span>";

  return p;
}

function renderItem(idx) {
  /** build bible */
  $('#bibleLink').empty();
  $('#bible').text(data[idx]["ref"]);
  $("#bibleLink").append("<a href='" + data[idx]["url"] + "' target='_blank'>" + data[idx]['url'] + "</a>");

  /** build bernard */
  $('#bernardLink').empty();
  var $bernard = $('#bernard');
  $bernard.empty();
  $bernard.append(buildTextMeta(data[idx]["textcontext"]["prev"], false, "prev"));
  $bernard.append(buildTextMeta(data[idx]["textdata"], true, "target"));
  $bernard.append(buildTextMeta(data[idx]["textcontext"]["next"], false, "next"));
  $('#bernardLink').append(
    "<a>" + data[idx]['sourceXml'] + ' - ' + data[idx]['id'] + "</a>");

  /** add annotation (if we are in review) */
  var highlight;
  var item;
  if (data[idx].hasOwnProperty("annotation")) {
    highlight = false;
    if (data[idx].annotation.prevSpan > 0) {
      var prevChildren = $('#prev').children();
      var prevSpan = Number(data[idx].annotation.prevSpan);
      for (var i=0; i<prevChildren.length;i++) {
	item = Number(prevChildren[i].getAttribute('data-item'));
	if (item === prevSpan) { highlight = true; }
	if (highlight) { $(prevChildren[i]).addClass("highlight"); }
      }
    }
    highlight = false;
    if (data[idx].annotation.nextSpan > 0) {
      var nextChildren = $('#next').children();
      var nextSpan = Number(data[idx].annotation.nextSpan);
      for (var i=0; i<nextChildren.length;i++) {
	item = Number(nextChildren[i].getAttribute('data-item'));
	if (item && item <= nextSpan) { highlight = true; }
	if (highlight) { $(nextChildren[i]).addClass("highlight"); }
	if (item && item === nextSpan) { highlight = false; }
      }
    }
    $('#target').addClass("highlight");
  }

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
    var selection = getSelectionData('bernard');
    console.log(selection);

    if (Object.keys(selection).length === 0) {
      /** warn on empty selection */
      $('#selectionWarning').modal();
      return;
    } else {
      /** send data to server */
      $.post(
	'saveAnnotation',
	{id: data[current]['id'],
	 sourceXml: data[current]['sourceXml'],
	 selectedText: selection['text'],
	 prevSpan: selection['span']['prev'],
	 nextSpan: selection['span']['next']}
      ).done(
	function() {
	  console.log("Correctly saved");
	}
      ).fail(
	function() {
	  console.log("Ooppsie while saving");
	}
      );
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
