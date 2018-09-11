% rebase('layout.tpl', title='Home Page', year=year)

<div class="container">

     <div class="jumbotron">
     <h1 class="display-4">Welcome!</h1>

     <p class="lead"> This is a tool to facilitate the annotation of spans in texts that match a particular property. </p>
     <p class="lead"> In our case, the property is whether the current text is a case of reuse with respect to another reference text. </p>
     <p class="lead"> As everybody knows, we will be annotating text reuse in Bernard, focusing on his borrowings from the Bible. </p>
     <p class="lead"> Before we can start we need to find some data to annotate. </p>
     <hr class="my-4">

     <div class="row my-4">
        <div class="col">

	%if len(opts) > 0:
	  %if not path:
	  <div class="alert alert-info" role="alert">
	    Please select your annotation file from the list below.
	  </div>
	  %end
	  <div class="list-group">
	    %for opt in opts:
	    <button type="button" data-path="{{opt}}" class="list-group-item list-group-item-action button-opt {{"active" if opt==path else ""}}">
	      {{opt}}
            </button>
	    %end
	  </div>
	  %if stats:
	  <h4 class="my-4">Progress</h4>
	  <ul class="list-group">
	    %for sourceXml, total, done, percentage in stats:
            <li class="list-group-item list-group-item-action flex-column align-items-start">
                <div class="d-flex w-100 justify-content-between">
                  <h5 class="mb-1">{{sourceXml}}</h5>
		  <small>{{done}}/{{total}} annotations done</small>
                </div>
                <p class="mb-1">
		  <div class="progress">
                    <div class="progress-bar" role="progressbar" style="width: {{percentage}}%;"></div>
                  </div>
		</p>
            </li>
	    %end
	  </ul>
	  %elif path:
	  <div class="alert alert-danger" role="alert">
	    Seems like <code>{{path}}</code> is not a valid annotation file.
	  </div>
	  %end

	%else:
	  <div class="alert alert-danger" role="alert">
	    Please add an annotation <code>.json</code> file to the path: <code>{{root}}</code>.
	  </div>

	%end
        </div>
     </div>

</div>

<script type="text/javascript" src="/static/scripts/index.js"></script>
