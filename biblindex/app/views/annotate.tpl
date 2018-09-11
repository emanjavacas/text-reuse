% rebase('layout.tpl', title='Annotate', year=year)


<div class="container">
  <div class="row">
    <div class="col">
      <nav class="navbar navbar-expand-md navbar-dark bg-dark">

	<ul class="navbar-nav">
          <li class="nav-item">
	    <form class="form-inline">
               <button id="skip" type="button" class="btn btn-sm btn-outline-info">Skip</button>  
            </form>
	  </li>
	</ul>

	<ul class="navbar-nav ml-auto">
          <li class="nav-item">
            <form class="form-inline">
              <button id="save" type="button" class="btn btn-sm btn-outline-success">Save</button>  
            </form>
          </li>
        </ul>
      </nav>
    </div>
  </div>

  <div class="row my-4"></div>

  <div class="card">
    <div class="card-body">
      <blockquote class="blockquote mb-0">
        <p id="bible">Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer posuere erat a ante.</p>
        <footer class="blockquote-footer" id="bibleLink"></footer>
      </blockquote>
    </div>
  </div>

  <hr/>

  <div class="card">
    <div class="card-body">
      <blockquote class="blockquote mb-0" id="bernard">
      </blockquote>
    </div>
  </div>


<div id="restartModal" class="modal" tabindex="-1" role="dialog">
  <div class="modal-dialog" role="document">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title">You've reached the end!</h5>
        <button type="button" class="close" data-dismiss="modal" aria-label="Close">
          <span aria-hidden="true">&times;</span>
        </button>
      </div>
      <div class="modal-body">
        <p>You have reached the end of the annotation file. Click on restart or refresh the page to go back to annotate the remaining instances. If there are none, you are done!</p>
      </div>
      <div class="modal-footer">
        <button id="restart" type="button" class="btn btn-primary">Restart</button>
      </div>
    </div>
  </div>
</div>

<div id="selectionWarning" class="modal" tabindex="-1" role="dialog">
  <div class="modal-dialog" role="document">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title">Nothing selected!</h5>
        <button type="button" class="close" data-dismiss="modal" aria-label="Close">
          <span aria-hidden="true">&times;</span>
        </button>
      </div>
      <div class="modal-body">
        <p>Annotation won't be saved for one of the following reasons:</p>
	<ul>
	<li>Nothing has been selected</li>
	<li>Selection contains text that doesn't belong to the target text</li>
	<li>Selection doesn't contain target text (highlighted text)</li>
	</ul>
	<p>Please try again after selecting a span in the Bernard panel (below panel).</p>
      </div>
    </div>
  </div>
</div>

</div>

<script>var data={{!data}}, current=0;</script>
<script type="text/javascript" src="static/scripts/annotate.js"></script>

