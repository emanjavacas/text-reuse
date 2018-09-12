<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }} - Text Reuse Span Annotation Tool</title>
    <link rel="stylesheet" type="text/css" href="/static/content/bootstrap.min.css" />
    <link rel="stylesheet" type="text/css" href="/static/content/site.css" />
    <script src="/static/scripts/jquery-1.10.2.js"></script>
    <script src="/static/scripts/bootstrap.js"></script>
  </head>

  <body>

    <nav class="navbar navbar-expand-md navbar-dark bg-dark">
      <ul class="navbar-nav mx-auto">
	<li>
	  <a class="navbar-brand mb-0 h1 text-center" href="/">Text Reuse Annotation Tool</a>
	</li>
	<li class="nav-item text-center">
	  %if active:
          <a class="nav-link" href="/annotate">Annotate</a>
	  %else:
	  <a class="nav-link disabled" onclick="return false;">Annotate</a>
	  %end
	</li>
	<li class="nav-item text-center">
	  %if active:
          <a class="nav-link" href="/review">Review</a>
	  %else:
	  <a class="nav-link disabled" onclick="return false;">Review</a>
	  %end
	</li>
      </ul>
    </nav>

    <div class="container">
      {{!base}}
      <footer class="navbar navbar-default navbar-fixed-bottom">
        <p class="muted">&copy; {{ year }} - Text Reuse Span Annotation Tool</p>
      </footer>
    </div>

  </body>
</html>
