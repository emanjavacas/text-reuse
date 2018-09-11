
$( document ).ready(function() {
  $(".button-opt").on('click', function(ev) {
    $.post(
      'register',
      {path: ev.target.dataset.path}
    ).done(
      function(){
	document.location.href="/";
      }
    ).fail(
      function(){
	console.log("Ooops, something went wrong!");
      }
    );
  });
});
